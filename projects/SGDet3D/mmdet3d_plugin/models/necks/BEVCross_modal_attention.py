import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmdet3d.models.builder import FUSION_LAYERS
from torchvision.utils import save_image
from mmcv.cnn import xavier_init
from mmcv.ops import MultiScaleDeformableAttention
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.ops import DeformConv2dPack as DeformConv2d

@FUSION_LAYERS.register_module()
class Cross_Modal_Fusion(nn.Module):
    def __init__(self, kernel_size=3, img_channels=256, rad_channels=384, out_channels=256):
        super(Cross_Modal_Fusion, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.att_img = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.att_radar = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.reduce_mixBEV = ConvModule(
                img_channels+rad_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)

    def forward(self, img_bev, radar_bev):
        img_avg_out = torch.mean(img_bev, dim=1, keepdim=True)
        img_max_out, _ = torch.max(img_bev, dim=1, keepdim=True)
        img_avg_max = torch.cat([img_avg_out, img_max_out], dim=1)
        img_att = self.att_img(img_avg_max)
        radar_avg_out = torch.mean(radar_bev, dim=1, keepdim=True)
        radar_max_out, _ = torch.max(radar_bev, dim=1, keepdim=True)
        radar_avg_max = torch.cat([radar_avg_out, radar_max_out], dim=1)
        radar_att = self.att_radar(radar_avg_max)
        img_bev = img_bev * radar_att
        radar_bev = radar_bev * img_att
        fusion_BEV = torch.cat([img_bev, radar_bev],dim=1)
        fusion_BEV = self.reduce_mixBEV(fusion_BEV)
        return fusion_BEV

@FUSION_LAYERS.register_module()
class Virtual_pts_BEV_Warp(nn.Module):
    def __init__(self, in_channel=384, 
                 d_model=256, 
                 nhead=8, 
                 num_levels=1, 
                 num_points=4, 
                 positional_encoding=None,
                 num_cross_layers=3,
                 num_self_layers=1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_levels = num_levels
        self.num_points = num_points
        self.in_channel = in_channel
        
        
        # input projection
        self.input_proj_query = nn.Conv2d(in_channel, d_model, kernel_size=1, stride=1, padding=0)
        self.input_proj_value = nn.Conv2d(in_channel, d_model, kernel_size=1, stride=1, padding=0)
        # Positional encodings for just query
        self.w = positional_encoding['row_num_embed']
        self.h = positional_encoding['col_num_embed']
        self.positional_encoding = build_positional_encoding(positional_encoding)

        # Multi-layer cross-attention and single-layer self-attention
        self.cross_attentions = nn.ModuleList([MultiScaleDeformableAttention(d_model, nhead, num_levels, num_points) for _ in range(num_cross_layers)])
        self.self_attentions = nn.ModuleList([MultiScaleDeformableAttention(d_model, nhead, num_levels, num_points) for _ in range(num_self_layers)])

        # Linear projection layers
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.LayerNorm(d_model // 2), nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, d_model // 4), nn.LayerNorm(d_model // 4), nn.ReLU(inplace=True),
            nn.Linear(d_model // 4, in_channel))
        
        self.init_weights()

    def generate_ref_2d(self, bs, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, self.h - 0.5, self.h, dtype=dtype, device=device), 
            torch.linspace(0.5, self.w - 0.5, self.w, dtype=dtype, device=device))
        ref_y = ref_y.flatten(0)[None] / self.h
        ref_x = ref_x.flatten(0)[None] / self.w
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
        return ref_2d.contiguous()
        
    def init_weights(self):
        """Initialize weights."""
        xavier_init(self.input_proj_query, distribution='uniform', bias=0)
        xavier_init(self.input_proj_value, distribution='uniform', bias=0)
        xavier_init(self.linear_query, distribution='uniform', bias=0)
        xavier_init(self.linear_value, distribution='uniform', bias=0)
        xavier_init(self.output_proj, distribution='uniform', bias=0)

    def forward(self, query, value):
        # Input projection and flattening
        query = self.input_proj_query(query)
        value = self.input_proj_value(value)
        B, C, H, W = query.shape
        query = query.flatten(2).permute(2, 0, 1)
        value = value.flatten(2).permute(2, 0, 1)
        
        # Positional encoding
        spatial_shapes = torch.tensor([[self.h, self.w]], device=query.device)
        level_start_index = torch.tensor([0], device=query.device)
        device, dtype = query.device, query.dtype
        pos = self.positional_encoding(torch.zeros((B, self.h, self.w), device=device).to(dtype)).to(dtype)
        pos = pos.flatten(2).permute(2, 0, 1).contiguous()

        # Linear projections
        query = self.linear_query(query).contiguous()
        value = self.linear_value(value).contiguous()

        # Multiple cross-attention layers
        for attention in self.cross_attentions:
            reference_points = self.generate_ref_2d(B, dtype, device)
            query = attention(
                query=query, 
                value=value, 
                query_pos=pos,
                reference_points=reference_points, 
                spatial_shapes=spatial_shapes, 
                level_start_index=level_start_index)

        # Multiple self-attention layer
        for attention in self.self_attentions:
            reference_points = self.generate_ref_2d(B, dtype, device)
            query = attention(
                query=query, 
                value=query,
                query_pos=pos, 
                reference_points=reference_points, 
                spatial_shapes=spatial_shapes, 
                level_start_index=level_start_index)

        # Output projection
        output = self.output_proj(query).permute(1, 2, 0).view(B, self.in_channel, H, W)

        return output.contiguous()

@FUSION_LAYERS.register_module()
class RadarBEVFusionWithDCN(nn.Module):
    def __init__(self, in_channels, kernel_size=5):
        super(RadarBEVFusionWithDCN, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        # DCN layers
        self.dcn1 = nn.Sequential(
            DeformConv2d(in_channels * 2, in_channels * 2, kernel_size=self.kernel_size, padding=self.padding, bias=False, deform_groups=2),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True))
        
        self.dcn2 = nn.Sequential(
            DeformConv2d(in_channels * 2, in_channels * 2, kernel_size=self.kernel_size, padding=self.padding, bias=False, deform_groups=2),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True))

        # SE Blocks
        self.se1 = SEBlock(in_channels * 2, reduction=16)
        self.se2 = SEBlock(in_channels * 2, reduction=16)
        
        # Reduce to original dimensions
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)) 

    def forward(self, bev1, bev2):
        # Concatenate BEV feature maps
        bev_concat = torch.cat((bev1, bev2), dim=1)

        # Deformable convolution and SE blocks stack
        out = self.dcn1(bev_concat)
        out = self.se1(out)
        out = self.dcn2(out)
        out = self.se2(out)

        # Reduce to original dimensions
        out = self.reduce_conv(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
if __name__=="__main__":
    # Virtual_pts_BEV_Warp
    # raw_radar_bev = torch.randn((2, 256, 160, 160))
    # vit_radar_bev = torch.randn((2, 384, 160, 160))
    # positional_encoding=dict(
    #         type='LearnedPositionalEncoding',
    #         num_feats=256//2,
    #         row_num_embed=160,
    #         col_num_embed=160),
    # fusion_module = Virtual_pts_BEV_Warp(in_channel=384, d_model=256, nhead=8, num_levels=1, num_points=4, positional_encoding=positional_encoding[0])
    # out = fusion_module(query=vit_radar_bev, value=raw_radar_bev)
    # print(out.shape, sum(p.numel() for p in fusion_module.parameters()))
    
    # RadarBEVFusionWithDCN
    bev1 = torch.randn((2, 256, 160, 160))
    bev2 = torch.randn((2, 256, 160, 160))
    fusion_module = RadarBEVFusionWithDCN(in_channels=256)
    fused_output = fusion_module(bev1, bev2)
    print(fused_output.shape, sum(p.numel() for p in fusion_module.parameters()))