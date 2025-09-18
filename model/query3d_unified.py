from copy import copy
from functools import partial
from modules.third_party.mask3d.position_embedding import PositionEmbeddingCoordsSine
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from data.datasets.constant import PromptType
import torch.nn.functional as F

from modules.build import build_module_by_name
from modules.utils import calc_pairwise_locs
from model.build import MODEL_REGISTRY, BaseModel
from optim.utils import no_decay_param_group
from accelerate.logging import get_logger
from torch.cuda.amp import autocast

logger = get_logger(__name__)

class CoordinateEncoder(nn.Module):
    def __init__(self, hidden_size, use_projection=True):
        super().__init__()
        self.pos_enc = PositionEmbeddingCoordsSine(pos_type="fourier", d_pos=hidden_size, gauss_scale=1.0, normalize=True)
        if use_projection:
            self.feat_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size))
    
    def forward(self, coords, input_range):
        with autocast(enabled=False):
            pos = self.pos_enc(coords, input_range=input_range).permute(0, 2, 1)
        if hasattr(self, 'feat_proj'):
            pos = self.feat_proj(pos)
        return pos
    
@MODEL_REGISTRY.register()
class Query3DUnified_GroundFlow(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        # record parameters
        self.cfg = cfg
        self.memories = cfg.model.memories #[pc,prompt]
        self.heads = cfg.model.heads
        self.use_offline_voxel_fts = cfg.model.get('use_offline_voxel_fts', False)
        self.use_offline_attn_mask = cfg.model.get('use_offline_attn_mask', False)
        self.inputs = self.memories[:]
        self.pairwise_rel_type = self.cfg.model.obj_loc.pairwise_rel_type
        self.spatial_dim = self.cfg.model.obj_loc.spatial_dim
        self.num_heads = self.cfg.model.unified_encoder.args.num_attention_heads
        self.skip_query_encoder_mask_pred = cfg.model.get('skip_query_encoder_mask_pred', False)
        # build prompt type
        self.prompt_types = ['txt', 'loc']
        # build feature encoder
        for input in self.inputs:
            if input == 'prompt':
                for prompt_type in self.prompt_types: # only text prompt for now
                    if prompt_type == 'loc':
                        continue
                    encoder = prompt_type + '_encoder'
                    setattr(self, encoder, build_module_by_name(cfg.model.get(encoder)))
            else:
                encoder = input + '_encoder'
                setattr(self, encoder, build_module_by_name(cfg.model.get(encoder)))
        # build location encoder
        dim_loc = self.cfg.model.obj_loc.dim_loc #6
        hidden_size = self.cfg.model.hidden_size
        self.dim_loc = dim_loc
        self.hidden_size = hidden_size
        if self.dim_loc > 3:
            self.coord_encoder = nn.Sequential(
                nn.Linear(3, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            self.box_encoder = nn.Sequential(
                nn.Linear(3, hidden_size),
                nn.LayerNorm(hidden_size),
            )
        else:
            self.coord_encoder = CoordinateEncoder(hidden_size)
        # build unified encoder    
        self.unified_encoder = build_module_by_name(self.cfg.model.unified_encoder)
        # build task head
        for head in self.heads:
            head = head + '_head'
            setattr(self, head, build_module_by_name(cfg.model.get(head)))
        
    def prompt_encoder(self, data_dict,i):
        prompt = data_dict['step_prompts'][:,i,:]
        prompt_pad_masks = data_dict['step_masks'][:,i,:]
        prompt_type = data_dict['prompt_type']
        prompt_feat = torch.zeros(prompt.shape + (self.hidden_size,), device=prompt.device)
        
        for type in self.prompt_types: #[txt:1]
            # get idx
            idx = prompt_type == getattr(PromptType, type.upper()) #[B]
            if idx.sum() == 0:
                continue
            input = prompt[idx] #[B,T]
            mask = prompt_pad_masks[idx]
            # encode
            if type == 'txt':
                encoder = self.txt_encoder
                feat = encoder(input.long(), mask)
            elif type == 'loc':
                print(idx)
                loc_prompts = input[:, :self.dim_loc]
                if self.dim_loc > 3:
                    feat = self.coord_encoder(loc_prompts[:, :3]).unsqueeze(1) + self.box_encoder(loc_prompts[:, 3:6]).unsqueeze(1)
                else:
                    feat = self.coord_encoder(loc_prompts[:, :3].unsqueeze(1), input_range=[data_dict['coord_min'][idx], data_dict['coord_max'][idx]])
                mask[:, 1:] = False
            else:
                raise NotImplementedError(f'{type} is not implemented')
            # put back to orignal prompt
            prompt_feat[idx] = feat
            prompt_pad_masks[idx] = mask
        return prompt_feat, prompt_pad_masks.logical_not()
        
    def forward(self, data_dict):
        input_dict = {}
        pre_query = None
        pre_query_masks = None
        # build query
        mask = data_dict['query_pad_masks'].logical_not() #Already logical not here
        query_locs = data_dict['query_locs'][:, :, :self.dim_loc] #[center,size]
        coord_min = data_dict['coord_min']
        coord_max = data_dict['coord_max']
        if self.dim_loc > 3:
            query_pos  = self.coord_encoder(query_locs[:, :, :3]) + self.box_encoder(query_locs[:, :, 3:6])
        else:
            query_pos = self.coord_encoder(query_locs[:, :, :3], input_range=[coord_min, coord_max])
        feat = torch.zeros_like(query_pos)
        pos = query_pos
        input_dict['query'] = (feat, mask, pos)
        # encode fts including point, voxel, image, and prompt -- in sg3d, only point and prompt will be used
        # the semantics of the attention mask in pytorch (True as masked) is the opposite as Huggingface Transformers (False as masked)  
        fts_locs = data_dict['seg_center'] #equalize obj_locs [center,size]
        if self.dim_loc > 3:
            fts_pos = self.coord_encoder(fts_locs[:, :, :3]) + self.box_encoder(fts_locs[:, :,  3:6])
        else:
            fts_pos = self.coord_encoder(fts_locs[:, :, :3], input_range=[coord_min, coord_max])
        if self.dim_loc > 3:
            fts_pos += self.box_encoder(fts_locs[:, :, 3:6])
        
        pc_feat = self.pc_encoder(obj_feats = data_dict['pc_seg_fts']) #obj_fts
        pc_mask = data_dict['pc_seg_pad_masks'].logical_not()
        pc_pos = fts_pos
        input_dict['pc'] = [pc_feat, pc_mask, pc_pos]
        
        if self.unified_encoder.spatial_selfattn:
            pairwise_locs = calc_pairwise_locs(query_locs[:, :, :3], None, 
                                           pairwise_rel_type=self.pairwise_rel_type, spatial_dist_norm=True,
                                           spatial_dim=self.spatial_dim)
            # print(pairwise_locs.shape) [B,N,N,5] spatial information
        else:
            pairwise_locs = None
            
        data_dict['ground_logits'] = None
        
        history_query = None
        for i in range(data_dict['step_prompts'].shape[1]): #[pc,prompt]
            promt_feat, prompt_mask, prompt_pos = None, None, None
            promt_feat, prompt_mask = self.prompt_encoder(data_dict,i)    
            input_dict['prompt'] = [promt_feat, prompt_mask, prompt_pos]


            
        # unified encoding                           
            query, query_masks, predictions_class, predictions_mask = self.unified_encoder(history_query, pre_query, pre_query_masks, input_dict, pairwise_locs)
            
            if history_query is not None: #Jh
                history_query = history_query + pre_query
                # history_query = history_query
            elif pre_query is not None: #Jt-1(hat)
                history_query = pre_query
            
            pre_query = query
            pre_query_masks = query_masks
        

            inputs = [query, data_dict['query_pad_masks']]
            logits = getattr(self, 'ground_head')(*inputs)
            
            if data_dict['ground_logits'] is None:
                data_dict['ground_logits'] = logits.unsqueeze(1)
            else:
                # Concatenate along the desired dimension (e.g., dim=1 for steps)
                data_dict['ground_logits'] = torch.cat((data_dict['ground_logits'], logits.unsqueeze(1)), dim=1)
            
        data_dict['og3d_logits'] = data_dict['ground_logits']
        data_dict['ground_label'] = data_dict["tgt_object_id"].squeeze(1)

    
        return data_dict
    
    def get_opt_params(self):
        def get_lr(cfg, default_lr):
            return default_lr if cfg is None or cfg.get("lr") is None else cfg.get("lr")

        optimizer_grouped_parameters = []
        for name, module in self._modules.items():
            module_cfg = self.cfg.model.get(name)
            lr = get_lr(module_cfg, self.cfg.solver.lr)
            if lr != self.cfg.solver.lr:
                print(f"Change lr from default {self.cfg.solver.lr} to {lr} for {name} module.")
            optimizer_grouped_parameters += no_decay_param_group(module.named_parameters(), lr, name=name)

        optimized_parameters = [p for group in optimizer_grouped_parameters for p in group['params']]
        assert len(optimized_parameters) == len(list(self.parameters())), "Some parameters are not optimized!"
        return optimizer_grouped_parameters
    
    def get_learnable_named_params(self):
        learnable_named_params = {}
        frozen_named_params = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                learnable_named_params.update({n: p})
            else:
                frozen_named_params.update({n: p})
        learnable_params_size = self.count_params(learnable_named_params.values())
        frozen_params_size = self.count_params(frozen_named_params.values())
        logger.info(
            f"Build PQ3D with {self.show_params_size(learnable_params_size+frozen_params_size)} parameters, " +
            f"{self.show_params_size(learnable_params_size)} learnable and " +
            f"{self.show_params_size(frozen_params_size)} frozen"
        )
       
        
        return learnable_named_params 
    

    
@MODEL_REGISTRY.register()
class Query3DUnified_Single(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        # record parameters
        self.cfg = cfg
        self.memories = cfg.model.memories #[pc,prompt]
        self.heads = cfg.model.heads
        self.use_offline_voxel_fts = cfg.model.get('use_offline_voxel_fts', False)
        self.use_offline_attn_mask = cfg.model.get('use_offline_attn_mask', False)
        self.inputs = self.memories[:]
        self.pairwise_rel_type = self.cfg.model.obj_loc.pairwise_rel_type
        self.spatial_dim = self.cfg.model.obj_loc.spatial_dim
        self.num_heads = self.cfg.model.unified_encoder.args.num_attention_heads
        self.skip_query_encoder_mask_pred = cfg.model.get('skip_query_encoder_mask_pred', False)
        # build prompt type
        self.prompt_types = ['txt', 'loc']
        # build feature encoder
        for input in self.inputs:
            if input == 'prompt':
                for prompt_type in self.prompt_types: # only text prompt for now
                    if prompt_type == 'loc':
                        continue
                    encoder = prompt_type + '_encoder'
                    setattr(self, encoder, build_module_by_name(cfg.model.get(encoder)))
            else:
                encoder = input + '_encoder'
                setattr(self, encoder, build_module_by_name(cfg.model.get(encoder)))
        # build location encoder
        dim_loc = self.cfg.model.obj_loc.dim_loc #6
        hidden_size = self.cfg.model.hidden_size
        self.dim_loc = dim_loc
        self.hidden_size = hidden_size
        if self.dim_loc > 3:
            self.coord_encoder = nn.Sequential(
                nn.Linear(3, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            self.box_encoder = nn.Sequential(
                nn.Linear(3, hidden_size),
                nn.LayerNorm(hidden_size),
            )
        else:
            self.coord_encoder = CoordinateEncoder(hidden_size)
        # build unified encoder    
        self.unified_encoder = build_module_by_name(self.cfg.model.unified_encoder)
        # build task head
        for head in self.heads:
            head = head + '_head'
            setattr(self, head, build_module_by_name(cfg.model.get(head)))
        
    def prompt_encoder(self, data_dict):
        prompt = data_dict['prompt']
        prompt_pad_masks = data_dict['prompt_pad_masks']
        prompt_type = data_dict['prompt_type']
        # print(prompt.shape,prompt_pad_masks.shape,prompt_type) [B,T],[B,T],1
        prompt_feat = torch.zeros(prompt.shape + (self.hidden_size,), device=prompt.device)
        for type in self.prompt_types: #[txt:1,loc:3]
            # get idx
            idx = prompt_type == getattr(PromptType, type.upper())
            if idx.sum() == 0:
                continue
            input = prompt[idx]
            mask = prompt_pad_masks[idx]
            # encode
            if type == 'txt':
                encoder = self.txt_encoder
                feat = encoder(input.long(), mask)
            elif type == 'loc':
                loc_prompts = input[:, :self.dim_loc]
                if self.dim_loc > 3:
                    feat = self.coord_encoder(loc_prompts[:, :3]).unsqueeze(1) + self.box_encoder(loc_prompts[:, 3:6]).unsqueeze(1)
                else:
                    feat = self.coord_encoder(loc_prompts[:, :3].unsqueeze(1), input_range=[data_dict['coord_min'][idx], data_dict['coord_max'][idx]])
                mask[:, 1:] = False
            else:
                raise NotImplementedError(f'{type} is not implemented')
            # put back to orignal prompt
            prompt_feat[idx] = feat
            prompt_pad_masks[idx] = mask
        return prompt_feat, prompt_pad_masks.logical_not()
        
    def forward(self, data_dict):
        input_dict = {}
        # build query
        mask = data_dict['query_pad_masks'].logical_not()
        query_locs = data_dict['query_locs'][:, :, :self.dim_loc]
        coord_min = data_dict['coord_min']
        coord_max = data_dict['coord_max']
        if self.dim_loc > 3:
            query_pos  = self.coord_encoder(query_locs[:, :, :3]) + self.box_encoder(query_locs[:, :, 3:6])
        else:
            query_pos = self.coord_encoder(query_locs[:, :, :3], input_range=[coord_min, coord_max])
        feat = torch.zeros_like(query_pos)
        pos = query_pos
        input_dict['query'] = (feat, mask, pos)
        # encode fts including point, voxel, image, and prompt -- in sg3d, only point and prompt will be used
        # the semantics of the attention mask in pytorch (True as masked) is the opposite as Huggingface Transformers (False as masked)  
        fts_locs = data_dict['seg_center']
        if self.dim_loc > 3:
            fts_pos = self.coord_encoder(fts_locs[:, :, :3]) + self.box_encoder(fts_locs[:, :,  3:6])
        else:
            fts_pos = self.coord_encoder(fts_locs[:, :, :3], input_range=[coord_min, coord_max])
        if self.dim_loc > 3:
            fts_pos += self.box_encoder(fts_locs[:, :, 3:6])
        for input in self.inputs:
            feat, mask, pos = None, None, None
            if input == 'prompt':
                feat, mask = self.prompt_encoder(data_dict)
                # print(feat.shape,mask.shape) #[B,T,H],[B,T]
            elif input == 'mv':
                feat = self.mv_encoder(obj_feats = data_dict['mv_seg_fts'])
                mask = data_dict['mv_seg_pad_masks'].logical_not()
                pos = fts_pos
            elif input == 'pc':
                feat = self.pc_encoder(obj_feats = data_dict['pc_seg_fts'])
                mask = data_dict['pc_seg_pad_masks'].logical_not()
                pos = fts_pos
            elif input == 'voxel':
                if self.use_offline_voxel_fts:
                    feat = self.voxel_encoder(data_dict['voxel_seg_fts'])
                    mask = data_dict['voxel_seg_pad_masks'].logical_not()
                else:
                    voxel_features = data_dict['voxel_features']
                    voxel_coordinates = data_dict['voxel_coordinates']
                    x = ME.SparseTensor(coordinates=voxel_coordinates, features=voxel_features[:, :-3], device=voxel_features.device)
                    voxel2segment = data_dict['voxel2segment']
                    feat = self.voxel_encoder(x, voxel2segment, max_seg=fts_locs.shape[1])
                    mask = data_dict['seg_pad_masks'].logical_not()
                    data_dict['voxel_feat'] = {'feat': feat[-1].detach().cpu(), 'mask': mask.detach().cpu()}
                pos = fts_pos
            else:
                raise NotImplementedError(f"Unknow input type: {input}")
            input_dict[input] = [feat, mask, pos]
        # build offline attention mask for guided mask training
        if self.use_offline_attn_mask:
            offline_attn_masks = data_dict['offline_attn_mask']
        else:
            offline_attn_masks = None
        # generate features for mask head
        seg_fts_for_match = []
        for input in self.inputs:
            if input in ['voxel', 'mv', 'pc']:
                feats = copy(input_dict[input][:])
                if isinstance(feats[0], list):
                    assert input == 'voxel'
                    feats[0] = feats[0][-1] # use the last scale of voxel features for segment matching
                seg_fts_for_match.append(feats)                 
        # build mask head
        if hasattr(self, 'mask_head'):
            mask_head_partial = partial(self.mask_head, seg_fts_for_match=seg_fts_for_match, seg_masks=data_dict['seg_pad_masks'].logical_not(),
                                        offline_attn_masks=offline_attn_masks, skip_prediction=self.skip_query_encoder_mask_pred)
        else:
            mask_head_partial = None
        # generate features for spatial attention
        if self.unified_encoder.spatial_selfattn:
            pairwise_locs = calc_pairwise_locs(query_locs[:, :, :3], None, 
                                           pairwise_rel_type=self.pairwise_rel_type, spatial_dist_norm=True,
                                           spatial_dim=self.spatial_dim)
        else:
            pairwise_locs = None
            
        # unified encoding                           
        query, predictions_class, predictions_mask = self.unified_encoder(input_dict, pairwise_locs, mask_head_partial)
        
        # task head
        for head in self.heads:
            if head == 'ground':
                inputs = [query, data_dict['query_pad_masks']]
                label = data_dict["tgt_object_id"]
                logits = getattr(self, head + '_head')(*inputs)
                data_dict[head + '_logits'] = logits
                data_dict['og3d_logits'] = logits
                data_dict[head + '_label'] = label.squeeze(1)
            elif head == 'generation':
                label = data_dict["response"]
                inputs = [query, data_dict['query_pad_masks']] + [label if self.training else None]
                logits = getattr(self, head + '_head')(*inputs)
                data_dict[head + '_logits'] = logits
                data_dict[head + '_label'] = label
            elif head == 'mask':
                if self.skip_query_encoder_mask_pred:
                    mask_head_partial = partial(self.mask_head, seg_fts_for_match=seg_fts_for_match, seg_masks=data_dict['seg_pad_masks'].logical_not(),
                                    offline_attn_masks=offline_attn_masks, skip_prediction=False)
                    predictions_class = []
                    predictions_mask = []
                pred_logits, pred_masks, _ = mask_head_partial(query=query)
                predictions_class.append(pred_logits)
                predictions_mask.append(pred_masks)
                data_dict['predictions_class'] = predictions_class
                data_dict['predictions_mask'] = predictions_mask
                continue
            else:
                raise NotImplementedError(f"Unknow head type: {head}")
       
        return data_dict

    def get_opt_params(self):
        def get_lr(cfg, default_lr):
            return default_lr if cfg is None or cfg.get("lr") is None else cfg.get("lr")

        optimizer_grouped_parameters = []
        for name, module in self._modules.items():
            module_cfg = self.cfg.model.get(name)
            lr = get_lr(module_cfg, self.cfg.solver.lr)
            if lr != self.cfg.solver.lr:
                print(f"Change lr from default {self.cfg.solver.lr} to {lr} for {name} module.")
            optimizer_grouped_parameters += no_decay_param_group(module.named_parameters(), lr, name=name)

        optimized_parameters = [p for group in optimizer_grouped_parameters for p in group['params']]
        assert len(optimized_parameters) == len(list(self.parameters())), "Some parameters are not optimized!"
        return optimizer_grouped_parameters
    
    def get_learnable_named_params(self):
        learnable_named_params = {}
        frozen_named_params = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                learnable_named_params.update({n: p})
            else:
                frozen_named_params.update({n: p})
        learnable_params_size = self.count_params(learnable_named_params.values())
        frozen_params_size = self.count_params(frozen_named_params.values())
        logger.info(
            f"Build PQ3D with {self.show_params_size(learnable_params_size+frozen_params_size)} parameters, " +
            f"{self.show_params_size(learnable_params_size)} learnable and " +
            f"{self.show_params_size(frozen_params_size)} frozen"
        )
        # logger.info(f"🧊 Frozen parameters: {list(frozen_named_params.keys())}")
        # logger.info(f"🔥 Tuned parameters: {list(learnable_named_params.keys())}")
        
        return learnable_named_params 
