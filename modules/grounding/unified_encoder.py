import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.build import GROUNDING_REGISTRY
from modules.layers.transformers import (CrossAttentionLayer,
                                         DualstreamLayer,
                                         TransformerDecoderLayer,
                                         Gated_CrossAttentionLayer,
                                         TransformerEncoderLayer,
                                         TransformerSpatialDecoderLayer)
from modules.utils import layer_repeat, calc_pairwise_locs
from modules.weights import _init_weights_bert
# from pytorch_forecasting import TemporalFusionTransformer


@GROUNDING_REGISTRY.register()
class EntitySpatialCrossEncoder(nn.Module):
    """
       spatial_dim: spatial feature dim, used to modify attention
       dim_loc:
    """

    def __init__(self, cfg, hidden_size=768, num_attention_heads=12, spatial_dim=5, num_layers=4, dim_loc=6,
                 pairwise_rel_type='center'):
        super().__init__()
        decoder_layer = TransformerSpatialDecoderLayer(hidden_size, num_attention_heads, dim_feedforward=2048,
                                                       dropout=0.1, activation='gelu',
                                                       spatial_dim=spatial_dim, spatial_multihead=True,
                                                       spatial_attn_fusion='cond')
        self.layers = layer_repeat(decoder_layer, num_layers)
        loc_layer = nn.Sequential(
            nn.Linear(dim_loc, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.loc_layers = layer_repeat(loc_layer, 1)
        self.pairwise_rel_type = pairwise_rel_type
        self.spatial_dim = spatial_dim
        self.spatial_dist_norm = True
        self.apply(_init_weights_bert)

    def forward(
            self, txt_embeds, txt_masks, obj_embeds, obj_locs, obj_masks,
            output_attentions=False, output_hidden_states=False, **kwargs
    ):
        pairwise_locs = calc_pairwise_locs(
            obj_locs[:, :, :3], obj_locs[:, :, 3:],
            pairwise_rel_type=self.pairwise_rel_type
        )

        out_embeds = obj_embeds
        for i, layer in enumerate(self.layers):
            query_pos = self.loc_layers[0](obj_locs)
            out_embeds = out_embeds + query_pos

            out_embeds, self_attn_matrices, cross_attn_matrices = layer(
                out_embeds, txt_embeds, pairwise_locs,
                tgt_key_padding_mask=obj_masks.logical_not(),
                memory_key_padding_mask=txt_masks.logical_not(),
            )

        return txt_embeds, out_embeds


@GROUNDING_REGISTRY.register()
class UnifiedSpatialCrossEncoder_Single(nn.Module):
    """
       spatial_dim: spatial feature dim, used to modify attention
       dim_loc:
    """

    def __init__(self, cfg, hidden_size=768, dim_feedforward=2048, num_attention_heads=12, num_layers=4, dim_loc=6):
        super().__init__()

        # unfied encoder
        unified_encoder_layer = TransformerEncoderLayer(hidden_size, num_attention_heads, dim_feedforward=dim_feedforward)
        self.unified_encoder = layer_repeat(unified_encoder_layer, num_layers)

        # loc layer
        loc_layer = nn.Sequential(
            nn.Linear(dim_loc, hidden_size),#dim_loc = 6 (xyzwhl)
            nn.LayerNorm(hidden_size),
        )
        self.loc_layers = layer_repeat(loc_layer, 1)

        # token embedding
        self.token_type_embeddings = nn.Embedding(2, hidden_size) #text token --0, ptc token -- 1

        self.apply(_init_weights_bert)

    def forward(
            self, txt_embeds, txt_masks, obj_embeds, obj_locs, obj_masks,
            output_attentions=False, output_hidden_states=False, **kwargs
    ):
        txt_len = txt_embeds.shape[1]
        obj_len = obj_embeds.shape[1]
        # print(obj_embeds.shape) [B,O,F(768)]

        for i, unified_layer in enumerate(self.unified_encoder):
            # add embeddings for points
            query_pos = self.loc_layers[0](obj_locs)
            pc_token_type_ids = torch.ones((obj_embeds.shape[0:2])).long().cuda()
            pc_type_embeds = self.token_type_embeddings(pc_token_type_ids)
            obj_embeds = obj_embeds + query_pos + pc_type_embeds #obj_emb + location of self + pc embed

            # add embeddings for languages
            lang_token_type_ids = torch.zeros((txt_embeds.shape[0:2])).long().cuda()
            lang_type_embeds = self.token_type_embeddings(lang_token_type_ids)
            txt_embeds = txt_embeds + lang_type_embeds #txt_embed+lang embed

            # fuse embeddings
            joint_embeds = torch.cat((txt_embeds, obj_embeds), dim=1)
            # print(f'txt_mask:{txt_masks},{txt_masks.shape},obj_masks:{obj_masks},{obj_masks.shape}')
            # txt_masks and obj_masks are used to mask out the padding text and object in collate
            joint_masks = torch.cat((txt_masks, obj_masks), dim=1)

            # transformer
            joint_embeds, self_attn_matrices = unified_layer(joint_embeds,
                                                             tgt_key_padding_mask=joint_masks.logical_not()) 
            #self-attention but considering the text and object embeddings are concat, similar to cross-att?

            # split
            txt_embeds, obj_embeds = torch.split(joint_embeds, [txt_len, obj_len], dim=1)
            # print(txt_embeds.shape,obj_embeds.shape,joint_embeds.shape)
        return txt_embeds, obj_embeds

@GROUNDING_REGISTRY.register() 
class UnifiedSpatialCrossEncoder_GroundFlow(nn.Module):

    def __init__(self, cfg, hidden_size=768, dim_feedforward=2048, num_attention_heads=12, num_layers=4, dim_loc=6):
        super().__init__()

        # unfied encoder
        unified_encoder_layer = TransformerEncoderLayer(hidden_size, num_attention_heads, dim_feedforward=dim_feedforward)
        self.unified_encoder = layer_repeat(unified_encoder_layer, num_layers)

        # History fusion encoder
        history_fusion_layer = CrossAttentionLayer(hidden_size, num_attention_heads, dim_feedforward=dim_feedforward)
        self.history_encoder = layer_repeat(history_fusion_layer, num_layers)
        # loc layer
        loc_layer = nn.Sequential(
            nn.Linear(dim_loc, hidden_size),#dim_loc = 6 (xyzwhl)
            nn.LayerNorm(hidden_size),
        )
        self.loc_layers = layer_repeat(loc_layer, 1)

        # token embedding
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.apply(_init_weights_bert)

    def forward(
            self, history_joint_features,pre_joint_embeds, pre_joint_masks, txt_embeds, txt_masks, obj_embeds, obj_locs, obj_masks,
            output_attentions=False, output_hidden_states=False, **kwargs
    ):
        txt_len = txt_embeds.shape[1]
        obj_len = obj_embeds.shape[1]


        for i, unified_layer in enumerate(self.unified_encoder):

            query_pos = self.loc_layers[0](obj_locs)
            pc_token_type_ids = torch.ones((obj_embeds.shape[0:2])).long().cuda()
            pc_type_embeds = self.token_type_embeddings(pc_token_type_ids)
            obj_embeds = obj_embeds + query_pos + pc_type_embeds #obj_emb + location of self + pc embed


            lang_token_type_ids = torch.zeros((txt_embeds.shape[0:2])).long().cuda()
            lang_type_embeds = self.token_type_embeddings(lang_token_type_ids)
            txt_embeds = txt_embeds + lang_type_embeds #txt_embed+lang embed


            joint_embeds = torch.cat((txt_embeds, obj_embeds), dim=1)

            joint_masks = torch.cat((txt_masks, obj_masks), dim=1)
        
            joint_embeds, self_attn_matrices = unified_layer(joint_embeds,
                                                             tgt_key_padding_mask=joint_masks.logical_not()) 

        
            txt_embeds, obj_embeds = torch.split(joint_embeds, [txt_len, obj_len], dim=1)
            
            
        if pre_joint_embeds is not None:

            if history_joint_features is not None:
                joint_embeds_normalized = F.normalize(joint_embeds, p=2, dim=-1)
                history_joint_features_normalized = F.normalize(history_joint_features, p=2, dim=-1)

                cosine_similarity = torch.bmm(joint_embeds_normalized, history_joint_features_normalized.transpose(1, 2))
                a = cosine_similarity.mean(dim=1, keepdim=True).transpose(1,2) 

                pre_joint_embeds = pre_joint_embeds + a*history_joint_features

            for i, history_encoder in enumerate(self.history_encoder):
                joint_embeds, cross_attn_matrices = history_encoder(joint_embeds,pre_joint_embeds,memory_key_padding_mask = pre_joint_masks.logical_not())
                
        txt_embeds, obj_embeds = torch.split(joint_embeds, [txt_len, obj_len], dim=1)
        return txt_embeds, obj_embeds, joint_embeds, joint_masks, pre_joint_embeds
 
