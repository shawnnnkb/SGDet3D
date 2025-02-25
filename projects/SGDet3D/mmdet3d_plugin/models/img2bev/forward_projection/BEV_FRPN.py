# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
import torch.utils.checkpoint as cp
from mmdet3d.models.builder import FUSION_LAYERS
from mmdet.models import weight_reduce_loss
from torchvision.utils import save_image
import copy

@FUSION_LAYERS.register_module()
class FRPN(BaseModule):
    r"""
    Args:
        in_channels (int): Channels of input feature.
        context_channels (int): Channels of transformed feature.
    """

    def __init__(
        self,
        in_channels=512,
        scale_factor=1,
        mask_thre=0.4,
        topk_rate_test = 0.01,
        loss_weight = 1.0
    ):
        super(FRPN, self).__init__()
        self.mask_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, 1, kernel_size=3, padding=1, stride=1),
            )
        self.upsample = nn.Upsample(scale_factor = scale_factor , mode ='bilinear', align_corners = True)
        self.dice_loss = FUSION_LAYERS.build(dict(type='CustomDiceLoss', use_sigmoid=True, loss_weight=1.))
        self.ce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.13]))  # From lss
        self.mask_thre = mask_thre
        self.topk_rate_test = topk_rate_test
        self.loss_weight = loss_weight

    def forward(self, input):
        """
        """
        bev_mask = self.mask_net(input)            
        bev_mask = self.upsample(bev_mask)
        return bev_mask
    
    def get_bev_mask_loss(self, gt_bev_mask, pred_bev_mask):
        bs, _, bev_h, bev_w = gt_bev_mask.shape
        b = gt_bev_mask.reshape(bs, bev_w * bev_h).permute(1, 0).to(torch.float)
        a = pred_bev_mask.reshape(bs, bev_w * bev_h).permute(1, 0)
        mask_ce_loss = self.ce_loss(a, b)*self.loss_weight
        mask_dc_loss = self.dice_loss(pred_bev_mask.reshape(bs, -1), gt_bev_mask.reshape(bs, -1))*self.loss_weight
        return dict(bev_mask_ce_loss=mask_ce_loss, 
                    bev_mask_dc_loss=mask_dc_loss)


# def reassign_bev_mask(all_zero_bev_idx, bev_mask, bev_mask_logit_sigmoid, grid_size, topk_rate_test):
#     bev_h, bev_w = grid_size[0], grid_size[1]
#     all_zero_bev_num = torch.sum(all_zero_bev_idx)
#     flattened_logits = bev_mask_logit_sigmoid[all_zero_bev_idx].view(all_zero_bev_num, -1)
#     topk_values, topk_indices = torch.topk(flattened_logits, int(bev_h*bev_w*topk_rate_test), dim=1)
#     mask_tmp = torch.zeros((all_zero_bev_num, 1, bev_h, bev_w), dtype=torch.bool)
#     for i in range(all_zero_bev_num):
#         mask_tmp.view(all_zero_bev_num, -1)[i, topk_indices[i]] = True
#     mask_tmp = mask_tmp.view((all_zero_bev_num, 1, bev_h, bev_w)).to(bev_mask.device)
#     bev_mask[all_zero_bev_idx] = mask_tmp
#     return bev_mask

def reassign_bev_mask(all_zero_bev_idx, bev_mask, bev_mask_logit_sigmoid, grid_size, topk_rate_test):
    bev_h, bev_w = grid_size[0], grid_size[1]
    all_zero_bev_num = torch.sum(all_zero_bev_idx)
    flattened_logits = bev_mask_logit_sigmoid[all_zero_bev_idx].view(all_zero_bev_num, -1)
    topk_values, topk_indices = torch.topk(flattened_logits, int(bev_h*bev_w*topk_rate_test), dim=1)
    for i in range(all_zero_bev_num):
        bev_mask.view(all_zero_bev_num, -1)[i, topk_indices[i]] = True
    bev_mask = bev_mask.view((all_zero_bev_num, 1, bev_h, bev_w))
    return bev_mask

def dice_loss(pred,
              target,
              weight=None,
              eps=1e-3,
              reduction='mean',
              avg_factor=None):
    """Calculate dice loss, which is proposed in
    `V-Net: Fully Convolutional Neural Networks for Volumetric
    Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """

    input = pred.reshape(pred.size()[0], -1)
    target = target.reshape(target.size()[0], -1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + eps
    c = torch.sum(target * target, 1) + eps
    d = (2 * a) / (b + c)
    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@FUSION_LAYERS.register_module()
class CustomDiceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 loss_weight=1.0,
                 eps=1e-3):
        """`Dice Loss, which is proposed in
        `V-Net: Fully Convolutional Neural Networks for Volumetric
         Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            eps (float): Avoid dividing by zero. Defaults to 1e-3.
        """

        super(CustomDiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate

    def forward(self,
                pred,
                target,
                weight=None,
                reduction_override=None,
                avg_factor=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            else:
                raise NotImplementedError

        loss = self.loss_weight * dice_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor)

        return loss