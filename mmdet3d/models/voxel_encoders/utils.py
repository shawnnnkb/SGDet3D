# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import build_norm_layer
from mmcv.runner import auto_fp16
from torch import nn
from torch.nn import functional as F


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num (torch.Tensor): Actual number of points in each voxel.
        max_num (int): Max number of points in each voxel

    Returns:
        torch.Tensor: Mask indicates which points are valid inside a voxel.
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class VFELayer(nn.Module):
    """Voxel Feature Encoder layer.

    The voxel encoder is composed of a series of these layers.
    This module do not support average pooling and only support to use
    max pooling to gather features inside a VFE.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
        max_out (bool): Whether aggregate the features of points inside
            each voxel and only return voxel features.
        cat_max (bool): Whether concatenate the aggregated features
            and pointwise features.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 max_out=True,
                 cat_max=True):
        super(VFELayer, self).__init__()
        self.fp16_enabled = False
        self.cat_max = cat_max
        self.max_out = max_out
        # self.units = int(out_channels / 2)

        self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    @auto_fp16(apply_to=('inputs'), out_fp32=True)
    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (torch.Tensor): Voxels features of shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.

        Returns:
            torch.Tensor: Voxel features. There are three mode under which the
                features have different meaning.
                - `max_out=False`: Return point-wise features in
                    shape (N, M, C).
                - `max_out=True` and `cat_max=False`: Return aggregated
                    voxel features in shape (N, C)
                - `max_out=True` and `cat_max=True`: Return concatenated
                    point-wise features in shape (N, M, C).
        """
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        voxel_count = inputs.shape[1]

        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        pointwise = F.relu(x)
        # [K, T, units]
        if self.max_out:
            aggregated = torch.max(pointwise, dim=1, keepdim=True)[0]
        else:
            # this is for fusion layer
            return pointwise

        if not self.cat_max:
            return aggregated.squeeze(1)
        else:
            # [K, 1, units]
            repeated = aggregated.repeat(1, voxel_count, 1)
            concatenated = torch.cat([pointwise, repeated], dim=2)
            # [K, T, 2 * units]
            return concatenated


class PFNLayer(nn.Module):
    """Pillar Feature Net Layer.

    The Pillar Feature Net is composed of a series of these layers, but the
    PointPillars paper results only used a single PFNLayer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
        last_layer (bool): If last_layer, there is no concatenation of
            features.
        mode (str): Pooling model to gather features inside voxels.
            Default to 'max'.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 last_layer=False,
                 mode='max'):

        super().__init__()
        self.fp16_enabled = False
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.norm = build_norm_layer(norm_cfg, self.units)[1]  # 需要归一化的维度为64
        self.linear = nn.Linear(in_channels, self.units, bias=False)

        assert mode in ['max', 'avg']
        self.mode = mode

    @auto_fp16(apply_to=('inputs'), out_fp32=True)
    def forward(self, inputs, num_voxels=None, aligned_distance=None):
        """Forward function.

        Args:
            inputs (torch.Tensor): Pillar/Voxel inputs with shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.
            num_voxels (torch.Tensor, optional): Number of points in each
                voxel. Defaults to None.
            aligned_distance (torch.Tensor, optional): The distance of
                each points to the voxel center. Defaults to None.

        Returns:
            torch.Tensor: Features of Pillars.
        """
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()  # N,C 或者 N C L
        x = F.relu(x)

        if self.mode == 'max':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = torch.max(x, dim=1, keepdim=True)[0]  # 0是数，1是索引
        elif self.mode == 'avg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = x.sum(
                dim=1, keepdim=True) / num_voxels.type_as(inputs).view(
                    -1, 1, 1)

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated

class PFNLayer_Radar(nn.Module):
    """Pillar Feature Net Layer.

    The Pillar Feature Net is composed of a series of these layers, but the
    PointPillars paper results only used a single PFNLayer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
        last_layer (bool): If last_layer, there is no concatenation of
            features.
        mode (str): Pooling model to gather features inside voxels.
            Default to 'max'.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 last_layer=False,
                 mode='max'):

        super().__init__()
        self.fp16_enabled = False
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        self.in_channels1 = 8
        self.in_channels2 = 2
        self.in_channels3 = 2
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units1 = out_channels // 2
        self.units2 = out_channels // 4
        self.units3 = out_channels // 4
        self.norm1 = build_norm_layer(norm_cfg, self.units1)[1]  # 需要归一化的维度为32
        self.norm2 = build_norm_layer(norm_cfg, self.units2)[1]# 需要归一化的维度为15
        self.norm3 = build_norm_layer(norm_cfg, self.units3)[1]# 需要归一化的维度为16
        self.linear1 = nn.Linear(self.in_channels1, self.units1, bias=False)
        self.linear2 = nn.Linear(self.in_channels2, self.units2, bias=False)
        self.linear3 = nn.Linear(self.in_channels3, self.units3, bias=False)
        assert mode in ['max', 'avg']
        self.mode = mode

    @auto_fp16(apply_to=('inputs'), out_fp32=True)
    def forward(self, inputs, num_voxels=None, aligned_distance=None):
        """Forward function.

        Args:
            inputs (torch.Tensor): Pillar/Voxel inputs with shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.
            num_voxels (torch.Tensor, optional): Number of points in each
                voxel. Defaults to None.
            aligned_distance (torch.Tensor, optional): The distance of
                each points to the voxel center. Defaults to None.

        Returns:
            torch.Tensor: Features of Pillars.
        """
        spatio_input = inputs.index_select(2,torch.tensor([0,1,2,5,6,7,8,9]).to(inputs.device))
        velocity_input = inputs.index_select(2,torch.tensor([3,10]).to(inputs.device))
        snr_input = inputs.index_select(2,torch.tensor([4,11]).to(inputs.device))
        x1 = self.linear1(spatio_input)
        x1 = self.norm1(x1.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()  # N,C 或者 N C L
        x2 = self.linear2(velocity_input)
        x2 = self.norm2(x2.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        x3 = self.linear3(snr_input)
        x3 = self.norm3(x3.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                             1).contiguous()
        x = torch.cat((x1,x2,x3),dim=-1)
        x = F.relu(x)

        if self.mode == 'max':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = torch.max(x, dim=1, keepdim=True)[0]  # 0是数，1是索引
        elif self.mode == 'avg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = x.sum(
                dim=1, keepdim=True) / num_voxels.type_as(inputs).view(
                    -1, 1, 1)

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated