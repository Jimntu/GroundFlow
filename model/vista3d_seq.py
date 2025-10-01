from modules.build import build_module
from model.build import MODEL_REGISTRY, BaseModel
from optim.utils import no_decay_param_group
from accelerate.logging import get_logger
from modules.layers.transformers import CrossAttentionLayer
import torch.nn.functional as F
import json
import torch
import torch.nn as nn
import os

logger = get_logger(__name__)

@MODEL_REGISTRY.register()
class Vista3DSeq_GroundFlow(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.lang_encoder = build_module("language", self.cfg.model.language)
        self.point_encoder = build_module("vision", self.cfg.model.vision)
        self.unified_encoder = build_module("grounding", self.cfg.model.grounding)
        self.head_list = self.cfg.model.heads.head_list
        for head in self.head_list:
            setattr(self, head, build_module("heads", getattr(self.cfg.model.heads, head)))


    def forward(self, data_dict):
        # prepare dict
        
        if 'cur_step' not in data_dict.keys():
            data_dict['cur_step'] = 1
            data_dict['total_steps'] = 1

        point_basic_features, point_features_pre, obj_cls_raw_logits = self.point_encoder(data_dict['obj_fts'].float(),
                                                                                          data_dict['obj_locs'],
                                                                                          data_dict['obj_pad_masks'],
                                                                                          data_dict['obj_pad_masks'],
                                                                                          None,
                                                                                          data_dict['cur_step'],
                                                                                          data_dict['total_steps'])
        data_dict['obj_cls_raw_logits'] = obj_cls_raw_logits

        history_joint_features = None
        pre_joint_features = None
        pre_joint_masks = None
        all_step_logits = {
            'txt_cls_logits': None,
            'obj_cls_post_logits': None,
            'obj_cls_pre_logits': None,
            'ground_logits': None
        }
        
        for i in range(data_dict['step_prompts'].shape[1]):
            lang_basic_features = self.lang_encoder(data_dict['step_prompts'][:,i,:].long(), data_dict['step_masks'][:,i,:].bool())

            language_fuse_feature, point_fuse_feature, joint_features, joint_masks, _ = self.unified_encoder(history_joint_features, pre_joint_features, pre_joint_masks, lang_basic_features, data_dict['step_masks'][:,i,:].bool(),
                                                                            point_basic_features, data_dict['obj_locs'],
                                                                            data_dict['obj_pad_masks'])
            
            
            # recurrent pipeline
            if history_joint_features is not None:
                history_joint_features = history_joint_features + pre_joint_features
            else:
                history_joint_features = pre_joint_features
            pre_joint_features = joint_features
            pre_joint_masks = joint_masks
            
            

            # task head
            txt_cls_logits, obj_cls_post_logits, obj_cls_pre_logits, ground_logits = self.ground_head(language_fuse_feature,
                                                                                                point_fuse_feature,
                                                                                                point_features_pre,
                                                                                                data_dict['obj_pad_masks'])
            
            logits_to_concatenate = {
                'txt_cls_logits': txt_cls_logits,
                'obj_cls_post_logits': obj_cls_post_logits,
                'obj_cls_pre_logits': obj_cls_pre_logits,
                'ground_logits': ground_logits
            }
            for key, current_logit in logits_to_concatenate.items():
                if all_step_logits[key] is None:
                    # Initialize the tensor in the first iteration
                    all_step_logits[key] = current_logit.unsqueeze(1)
                else:
                    # Concatenate along the desired dimension (e.g., dim=1 for steps)
                    all_step_logits[key] = torch.cat((all_step_logits[key], current_logit.unsqueeze(1)), dim=1)

        data_dict.update(all_step_logits)
        data_dict['ground_label'] = data_dict['tgt_object_id'].squeeze(1)

        return data_dict

    def get_opt_params(self):
        def get_lr(cfg, default_lr):
            return default_lr if cfg.get("lr") is None else cfg.get("lr") #1e-4

        optimizer_grouped_parameters = []
        optimizer_grouped_parameters += no_decay_param_group(self.lang_encoder.named_parameters(),
                                                             get_lr(self.cfg.model.language, self.cfg.solver.lr))
        optimizer_grouped_parameters += no_decay_param_group(self.point_encoder.named_parameters(),
                                                             get_lr(self.cfg.model.vision, self.cfg.solver.lr))
        optimizer_grouped_parameters += no_decay_param_group(self.unified_encoder.named_parameters(),
                                                             get_lr(self.cfg.model.grounding, self.cfg.solver.lr))
        if "ground_head" in self.head_list:
            optimizer_grouped_parameters += no_decay_param_group(
                self.ground_head.named_parameters(), get_lr(self.cfg.model.heads.ground_head, self.cfg.solver.lr)
            )
        if "qa_head" in self.head_list:
            optimizer_grouped_parameters += no_decay_param_group(
                self.qa_head.named_parameters(), get_lr(self.cfg.model.heads.qa_head, self.cfg.solver.lr)
            )
        if "pretrain_head" in self.head_list:
            optimizer_grouped_parameters += no_decay_param_group(
                self.pretrain_head.named_parameters(), get_lr(self.cfg.model.heads.pretrain_head, self.cfg.solver.lr)
            )
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
            f"Build {self.__class__.__name__} with {self.show_params_size(learnable_params_size+frozen_params_size)} parameters, " +
            f"{self.show_params_size(learnable_params_size)} learnable and " +
            f"{self.show_params_size(frozen_params_size)} frozen"
        )
    
        
        return learnable_named_params 




    
@MODEL_REGISTRY.register()
class Vista3DSeq_Single(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.lang_encoder = build_module("language", self.cfg.model.language)
        self.point_encoder = build_module("vision", self.cfg.model.vision)
        self.unified_encoder = build_module("grounding", self.cfg.model.grounding)
        self.head_list = self.cfg.model.heads.head_list
        for head in self.head_list:
            setattr(self, head, build_module("heads", getattr(self.cfg.model.heads, head)))

    def forward(self, data_dict):
        # prepare dict
        if 'cur_step' not in data_dict.keys():
            data_dict['cur_step'] = 1
            data_dict['total_steps'] = 1
        # basic feature extracter, padding tokens are masked false
        # prompt is tokenized instruction (task info + prev steps), prompt_pad_masks is torch.ones((len(prompt))).bool()
        # print(data_dict['prompt'].shape,data_dict['prompt_pad_masks'].shape)  ----> Already padding for each batch
    
        lang_basic_features = self.lang_encoder(data_dict['prompt'].long(), data_dict['prompt_pad_masks'].bool())
        
        #obj_fts are normalized and centering xyz,rgb, obj_locs are torch.cat([center, size], dim=1), obj_pad_masks are torch.ones(len(obj))
        point_basic_features, point_features_pre, obj_cls_raw_logits = self.point_encoder(data_dict['obj_fts'].float(),
                                                                                          data_dict['obj_locs'],
                                                                                          data_dict['obj_pad_masks'],
                                                                                          data_dict['obj_pad_masks'],
                                                                                          None,
                                                                                          data_dict['cur_step'],
                                                                                          data_dict['total_steps'])
        # point_basic_features is features after spatial transformer, point_features_pre_spatial is point features before spatial reasoning
        # obj_cla_raw_logits are the predicted class of the object of size  B, N, O

        # unifed language entity transformer
        language_fuse_feature, point_fuse_feature = self.unified_encoder(lang_basic_features, data_dict['prompt_pad_masks'].bool(),
                                                                         point_basic_features, data_dict['obj_locs'],
                                                                         data_dict['obj_pad_masks'])

        # Use language_fuse_feature and point_fuse_feature for contrastive pretraining

        data_dict['obj_cls_raw_logits'] = obj_cls_raw_logits
        # task head
        if getattr(self, "ground_head", None) is not None:
            txt_cls_logits, obj_cls_post_logits, obj_cls_pre_logits, og3d_logits = self.ground_head(language_fuse_feature,
                                                                                                point_fuse_feature,
                                                                                                point_features_pre,
                                                                                                data_dict['obj_pad_masks'])
            data_dict['txt_cls_logits'] = txt_cls_logits# [B,O]
            data_dict['obj_cls_post_logits'] = obj_cls_post_logits #[B,N,O]
            data_dict['obj_cls_pre_logits'] = obj_cls_pre_logits #[B,N,O]
            data_dict['og3d_logits'] = og3d_logits #[B,N] N depends on the number of the objects in this scan
            data_dict['ground_logits']  = og3d_logits #[B,N]
            data_dict['ground_label'] = data_dict['tgt_object_id'].squeeze(1)
        if getattr(self, "qa_head", None) is not None:
            answer_scores = self.qa_head(point_fuse_feature, data_dict['obj_masks'], language_fuse_feature,
                                         data_dict['txt_masks'])
            data_dict['answer_scores'] = answer_scores
        if getattr(self, "pretrain_head", None) is not None:
            txt_lm_cls_logits = self.pretrain_head(language_fuse_feature)
            data_dict['txt_lm_cls_logits'] = txt_lm_cls_logits

        
        return data_dict

    def get_opt_params(self):
        def get_lr(cfg, default_lr):
            return default_lr if cfg.get("lr") is None else cfg.get("lr") #1e-4

        optimizer_grouped_parameters = []
        optimizer_grouped_parameters += no_decay_param_group(self.lang_encoder.named_parameters(),
                                                             get_lr(self.cfg.model.language, self.cfg.solver.lr))
        optimizer_grouped_parameters += no_decay_param_group(self.point_encoder.named_parameters(),
                                                             get_lr(self.cfg.model.vision, self.cfg.solver.lr))
        optimizer_grouped_parameters += no_decay_param_group(self.unified_encoder.named_parameters(),
                                                             get_lr(self.cfg.model.grounding, self.cfg.solver.lr))
        if "ground_head" in self.head_list:
            optimizer_grouped_parameters += no_decay_param_group(
                self.ground_head.named_parameters(), get_lr(self.cfg.model.heads.ground_head, self.cfg.solver.lr)
            )
        if "qa_head" in self.head_list:
            optimizer_grouped_parameters += no_decay_param_group(
                self.qa_head.named_parameters(), get_lr(self.cfg.model.heads.qa_head, self.cfg.solver.lr)
            )
        if "pretrain_head" in self.head_list:
            optimizer_grouped_parameters += no_decay_param_group(
                self.pretrain_head.named_parameters(), get_lr(self.cfg.model.heads.pretrain_head, self.cfg.solver.lr)
            )
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
            f"Build {self.__class__.__name__} with {self.show_params_size(learnable_params_size+frozen_params_size)} parameters, " +
            f"{self.show_params_size(learnable_params_size)} learnable and " +
            f"{self.show_params_size(frozen_params_size)} frozen"
        )
        # logger.info(f"🧊 Frozen parameters: {list(frozen_named_params.keys())}")
        # logger.info(f"🔥 Tuned parameters: {list(learnable_named_params.keys())}")
        
        return learnable_named_params 
