# ------------------------------------------------------------------------
# DFA3D
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the IDEA License, Version 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from BEVFormer (https://github.com/fundamentalvision/BEVFormer)
# Copyright (c) fundamentalvision. All rights reserved
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
#  Modified by Hongyang Li
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmcv.cnn import xavier_init
from mmdet3d.models.builder import FUSION_LAYERS
from mmcv.cnn.bricks.transformer import TRANSFORMER_LAYER_SEQUENCE
from torch.nn.init import normal_
from .deformable_self_attention import DeformSelfAttention
from .deformable_cross_attention import MSDeformableAttention3D
from abc import abstractmethod, ABCMeta
from torchvision.utils import save_image

@FUSION_LAYERS.register_module()
class PerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_cams_embeds=True,
                 use_level_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(PerceptionTransformer, self).__init__(**kwargs)
        self.encoder = FUSION_LAYERS.build(encoder) if encoder is not None else None
        self.decoder = FUSION_LAYERS.build(decoder) if decoder is not None else None
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_cams_embeds = use_cams_embeds
        self.use_level_embeds = use_level_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        if self.use_level_embeds:
            self.level_embeds = nn.Parameter(torch.Tensor(
                self.num_feature_levels, self.embed_dims))
        
        if self.use_cams_embeds:
            self.cams_embeds = nn.Parameter(
                torch.Tensor(self.num_cams, self.embed_dims))
            
        if self.decoder is not None:
            self.reference_points = nn.Linear(self.embed_dims, 3)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, DeformSelfAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        if self.use_level_embeds: normal_(self.level_embeds)
        if self.use_cams_embeds: normal_(self.cams_embeds)
        if self.decoder is not None:
            xavier_init(self.reference_points, distribution='uniform', bias=0.)

    def get_bev_features(self):
        "please instantiate in child class"
        pass
    
    def diffuse_bev_features(self):
        "please instantiate in child class"
        pass
    
    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        B, C, H, W = bev_pos.shape
        bev_pos = bev_pos.flatten(2).permute(0, 2, 1)
        bev_embed = bev_queries + bev_pos  # B, H*W, embed_dims
        # bev_embed = self.get_bev_features(
        #     mlvl_feats,
        #     bev_queries,
        #     bev_h,
        #     bev_w,
        #     grid_length=grid_length,
        #     bev_pos=bev_pos,
        #     prev_bev=prev_bev,
        #     **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = bev_queries.shape[0]
        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out
    
@FUSION_LAYERS.register_module()
class PerceptionTransformer_DFA3D(PerceptionTransformer):
    def get_bev_features(
            self,
            imgs,
            mlvl_feats,
            mlvl_dpt_dists,
            bev_queries,
            bev_h,
            bev_w,
            bev_coords,
            ref_bev,
            unmasked_idx,
            cam_params = None,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            img_metas=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """

        bs = mlvl_feats[0].size(0)
        # bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        if bev_queries.dim() == 2:
            bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1) # N bs C

        unmasked_bev_queries = bev_queries[bev_coords[unmasked_idx, -1], :, :]
        unmasked_bev_bev_pos = bev_pos[bev_coords[unmasked_idx, -1], :, :]

        unmasked_ref_bev = ref_bev[bev_coords[unmasked_idx, -1], :]
        unmasked_ref_bev = unmasked_ref_bev.unsqueeze(0).unsqueeze(0).to(unmasked_bev_queries.device)

        feat_flatten = []
        dpt_dist_flatten = []
        spatial_shapes = []
        for lvl, (feat, dpt_dist) in enumerate(zip(mlvl_feats, mlvl_dpt_dists)):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            dpt_dist = dpt_dist.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
            dpt_dist_flatten.append(dpt_dist)

        feat_flatten = torch.cat(feat_flatten, 2)
        dpt_dist_flatten = torch.cat(dpt_dist_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
        # level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        level_start_index = torch.cat((spatial_shapes.cpu().new_zeros((1,)), spatial_shapes.cpu().prod(1).cumsum(0)[:-1])).to(bev_pos.device)
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        dpt_dist_flatten = dpt_dist_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.encoder(
            imgs, 
            unmasked_bev_queries,
            feat_flatten,
            feat_flatten,
            value_dpt_dist=dpt_dist_flatten,
            ref_bev=unmasked_ref_bev,
            bev_h=bev_h,
            bev_w=bev_w,
            query_pos=unmasked_bev_bev_pos, # all for query if sparse
            key_pos=bev_pos, # all for key
            bev_pos=unmasked_bev_bev_pos, # not used actually
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            cam_params=cam_params,
            img_metas=img_metas,
            prev_bev=None,
            shift=None,
            **kwargs
        )

        return bev_embed
    
    def diffuse_vox_features(
            self,
            imgs,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            ref_bev,
            bev_coords,
            unmasked_idx,
            cam_params = None,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            img_metas=None,
            prev_bev=None,
            **kwargs):
        """
        diffuse voxel features.
        """

        bs = mlvl_feats[0].size(0)
        #  bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1) 
        if bev_pos is not None:
            bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        unmasked_ref_bev = ref_bev[bev_coords[unmasked_idx, -1], :]
        unmasked_ref_bev = unmasked_ref_bev.unsqueeze(0).unsqueeze(0).to(bev_queries.device)
        
        bev_embed = self.encoder(
            imgs,
            bev_queries,
            None,
            None,
            value_dpt_dist=None,
            ref_bev=unmasked_ref_bev,
            bev_h=bev_h,
            bev_w=bev_w,
            query_pos=bev_pos,  # all for query (not used)
            key_pos=bev_pos, # all for key (not used)
            bev_pos=bev_pos,  # all (self-attn already use bev_pos)
            spatial_shapes=None,
            level_start_index=None,
            cam_params=cam_params,
            img_metas=img_metas,
            prev_bev=None,
            shift=None,
            **kwargs
        ) 
        
        return bev_embed