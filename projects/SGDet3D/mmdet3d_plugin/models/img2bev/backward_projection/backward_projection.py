# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE


# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import torch
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet3d.models.builder import FUSION_LAYERS
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.core import multi_apply

@FUSION_LAYERS.register_module()
class BackwardProjection(BaseModule):
    def __init__(self,
                bev_h_=100,
                bev_w_=100,
                grid_config=None,
                data_config=None,
                point_cloud_range=None,
                embed_dims=128,
                cross_transformer=None,
                self_transformer=None,
                positional_encoding=None,
                mlp_prior=True,
                **kwargs):
        super().__init__()
        
        self.mlp_prior = mlp_prior
        self.bev_h = bev_h_
        self.bev_w = bev_w_
        self.real_h = point_cloud_range[3] - point_cloud_range[0]
        self.real_w = point_cloud_range[4] - point_cloud_range[1]
        # assert self.real_h / self.bev_h == grid_config['xbound'][2]
        # assert self.real_w / self.bev_w == grid_config['ybound'][2]
        self.embed_dims = embed_dims
        
        self.data_config = data_config
        self.grid_config = grid_config
        self.point_cloud_range = point_cloud_range
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.cross_transformer = FUSION_LAYERS.build(cross_transformer) if cross_transformer is not None else None
        self.self_transformer = FUSION_LAYERS.build(self_transformer) if self_transformer is not None else None
        self.grid_length = (self.real_h / self.bev_h, self.real_w / self.bev_w)
        self.bev_coords, self.ref_bev = self.get_voxel_indices()
        if mlp_prior:
            self.mlp_prior = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims//2),
            nn.LayerNorm(self.embed_dims//2),
            nn.LeakyReLU(),
            nn.Linear(self.embed_dims//2, self.embed_dims)
            )
        else:
            self.mlp_prior = None
            self.mask_embed = nn.Embedding(1, self.embed_dims)
    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        if self.cross_transformer is not None: self.cross_transformer.init_weights()
        if self.self_transformer is not None: self.self_transformer.init_weights()
    
    def get_voxel_indices(self):
        # find specific bev_coords via idx, because we reshape in transformer and hard to find accordingly xy 
        xv, yv = torch.meshgrid(torch.arange(self.bev_h), torch.arange(self.bev_w))
        idx = torch.arange(self.bev_h * self.bev_w)
        bev_coords = torch.cat([xv.reshape(-1, 1), yv.reshape(-1, 1), idx.reshape(-1, 1)], dim=-1)

        ref_bev = torch.cat(
            [(xv.reshape(-1, 1) + 0.5) / self.bev_h, 
             (yv.reshape(-1, 1) + 0.5) / self.bev_w], dim=-1)

        return bev_coords, ref_bev    
    def forward(self, imgs, mlvl_feats, proposal, cam_params, lss_bev=None, img_metas=None, mlvl_dpt_dists=None, backward_bev_mask_logit=None):
        # initialization
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype, device = mlvl_feats[0].dtype, mlvl_feats[0].device
        
        # depth estimation would bring benefit, here is intialization for queries
        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        if lss_bev is not None:
            bev_mask_logit_flatten = backward_bev_mask_logit.flatten(2).permute(2, 0, 1)
            lss_bev_flatten = lss_bev.flatten(2).permute(2, 0, 1)
            bev_queries = bev_queries + lss_bev_flatten
            bev_feats_identity = bev_mask_logit_flatten*lss_bev_flatten.clone()
        bev_queries_list = [bev_queries[:,i:i+1,:] for i in range(bs)]
        
        # FRPN force model to focus more on foregroud objects
        if proposal is not None:
            proposal_list = [proposal[i:i+1] for i in range(bs)]
            unmasked_idx_list = [torch.nonzero(p.reshape(-1) > 0).view(-1) for p in proposal_list]
            masked_idx_list = [torch.nonzero(p.reshape(-1) == 0).view(-1) for p in proposal_list]

        # Generate bev postional embeddings for cross and self attention
        bev_pos_cross_attn = self.positional_encoding(torch.zeros((1 , self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)).to(dtype)
        bev_pos_self_attn  = self.positional_encoding(torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)).to(dtype)
        bev_coords, ref_bev = self.bev_coords.clone(), self.ref_bev.clone()

        # only support batchsize = 1, because unmasked_idx_list is different among different bs
        seed_feats_list = multi_apply(self.cross_transformer.get_bev_features, 
            [imgs[i:i+1] for i in range(bs)],
            [[mlvl_feats[0][i:i+1]] for i in range(bs)],      # multi-level feature map list of [1,N,C,H,W]
            [[mlvl_dpt_dists[0][i:i+1]] for i in range(bs)],  # multi-level depth map list of [1,N,D,H,W]
            bev_queries_list,   # list of len(bs), each of [bev_w*bev_h, 1, embed_dims]
            [self.bev_h for _ in range(bs)], # list of len(bs), each a int
            [self.bev_w for _ in range(bs)], # list of len(bs), each a int
            [bev_coords for _ in range(bs)], # list of len(bs), each of [bev_w*bev_h, 3]
            [ref_bev for _ in range(bs)], # list of len(bs), each of [bev_w*bev_h, 2]
            unmasked_idx_list, # list of len(bs), each a tensor indicating unmasked queries
            cam_params, # list of len(bs), each is a list of all bs' cam_params
            [self.grid_length for _ in range(bs)], # list of len(bs), each is [bev_h, bev_w]
            [bev_pos_cross_attn for _ in range(bs)], # list of len(bs), each is [1, C, H, W] according to [bev_h, bev_w]
            img_metas) # list if img_meta, each is very complex
        
        # Use lss_bev from forward projection with mlp for padding or just mask_embed
        seed_feats_list = seed_feats_list[0]
        pad_bev_feats_list = []
        for i in range(bs):
            bev_feats = torch.empty((self.bev_h, self.bev_w, self.embed_dims), device=bev_queries.device)
            bev_feats_flatten = bev_feats.reshape(-1, self.embed_dims)
            # bev_feats_flatten = lss_bev_flatten[:,i:i+1,:].reshape(-1, self.embed_dims)
            unmasked_idx = unmasked_idx_list[i]
            masked_idx = masked_idx_list[i]
            bev_feats_flatten[bev_coords[unmasked_idx, -1], :] = seed_feats_list[i]
            if self.mlp_prior is None:
                bev_feats_flatten[bev_coords[masked_idx, -1], :] = self.mask_embed.weight.view(1, self.embed_dims).expand(masked_idx.shape[0], self.embed_dims).to(dtype)
            else:
                bev_feats_flatten[bev_coords[masked_idx, -1], :] = self.mlp_prior(lss_bev_flatten[masked_idx, i, :])
            pad_bev_feats = bev_feats_flatten.unsqueeze(1).to(dtype) + bev_feats_identity[:, i, :].unsqueeze(1)
            pad_bev_feats_list.append(pad_bev_feats)
        bev_feats_flatten = torch.cat(pad_bev_feats_list, dim=1).contiguous() # [num_queries, bs, dims]
        
        # combined to do self attention to get bev features
        final_bev_feats = self.self_transformer.diffuse_vox_features(
            imgs=imgs,
            mlvl_feats=mlvl_feats,
            bev_queries=bev_feats_flatten,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            ref_bev=ref_bev,
            bev_coords=bev_coords,
            unmasked_idx=unmasked_idx,
            grid_length=self.grid_length,
            bev_pos=bev_pos_self_attn,
            img_metas=img_metas,
            prev_bev=None,
            cam_params=cam_params)

        return final_bev_feats

