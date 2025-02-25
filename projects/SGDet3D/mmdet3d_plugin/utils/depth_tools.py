import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torchvision.utils import save_image
def get_downsample_depths_torch(depth, down, processing='min'):
    B, C, H, W = depth.shape
    depth = depth.view(B, H//down, down, W//down, down, 1)
    depth = depth.permute(0, 1, 3, 5, 2, 4).contiguous()
    depth = depth.view(-1, down * down)
    depth_tmp = torch.where(depth == 0.0, 1e5 * torch.ones_like(depth), depth)
    if processing == 'min': 
        depth = torch.min(depth_tmp, dim=-1).values
    if processing == 'max': 
        depth = torch.max(depth_tmp, dim=-1).values
    if processing == 'mean': 
        depth = torch.mean(depth_tmp, dim=-1)
    depth = depth.view(B, C, H//down, W//down)
    return depth
    
def generate_guassian_depth_target(depth, stride, cam_depth_range, constant_std=None):
    depth = depth.flatten(0, 1)
    B, tH, tW = depth.shape
    kernel_size = stride
    center_idx = kernel_size * kernel_size // 2
    H = tH // stride
    W = tW // stride
    
    unfold_depth = F.unfold(depth.unsqueeze(1), kernel_size, dilation=1, padding=0, stride=stride) # B, Cxkxk, HxW, here C=1
    unfold_depth = unfold_depth.view(B, -1, H, W).permute(0, 2, 3, 1).contiguous() # BN, H, W, kxk
    valid_mask = (unfold_depth != 0) # BN, H, W, kxk
    
    if constant_std is None:
        valid_mask_f = valid_mask.float() # BN, H, W, kxk
        valid_num = torch.sum(valid_mask_f, dim=-1) # BN, H, W
        valid_num[valid_num == 0] = 1e10
        
        mean = torch.sum(unfold_depth, dim=-1) / valid_num
        var_sum = torch.sum(((unfold_depth - mean.unsqueeze(-1))**2) * valid_mask_f, dim=-1) # BN, H, W
        std_var = torch.sqrt(var_sum / valid_num)
        std_var[valid_num == 1] = 1 # set std_var to 1 when only one point in patch
    else:
        std_var = torch.ones((B, H, W)).type_as(depth).float() * constant_std

    unfold_depth[~valid_mask] = 1e10
    min_depth = torch.min(unfold_depth, dim=-1)[0] # BN, H, W, min_depth in stridexstride block
    loss_valid_mask = ~(min_depth == 1e10)
    min_depth[min_depth == 1e10] = 0
    
    # x in raw depth 
    x = torch.arange(cam_depth_range[0] - cam_depth_range[2] / 2, cam_depth_range[1], cam_depth_range[2])
    # normalized by intervals
    dist = Normal(min_depth / cam_depth_range[2], std_var / cam_depth_range[2]) # BN, H, W, D
    # dist = Normal(min_depth, std_var / cam_depth_range[2]) # BN, H, W, D
    cdfs = []
    for i in x:
        cdf = dist.cdf(i)
        cdfs.append(cdf)
    
    cdfs = torch.stack(cdfs, dim=-1)
    depth_dist = cdfs[..., 1:] - cdfs[...,:-1]

    return depth_dist, min_depth

def sobel_operator(img):
    """
    Apply Sobel operator to compute image gradients.
    
    Args:
        img: torch.Tensor, the input image with shape (B, C, H, W)
    
    Returns:
        grad: torch.Tensor, the computed gradient magnitude with shape (B, 1, H, W)
    """
    sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).to(img.device).view(1, 1, 3, 3)
    sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).to(img.device).view(1, 1, 3, 3)

    grad_x = F.conv2d(img, sobel_x, padding=1, groups=img.shape[1])
    grad_y = F.conv2d(img, sobel_y, padding=1, groups=img.shape[1])
    grad = torch.sqrt(grad_x**2 + grad_y**2)
    
    return grad
def edge_aware_smoothness_loss(img, depth):
    """
    Edge-aware smoothness loss.
    
    Args:
        img: torch.Tensor, the input image with shape (B, C, H, W)
        depth: torch.Tensor, the estimated depth map with shape (B, 1, H, W)
    
    Returns:
        torch.Tensor, the computed edge-aware smoothness loss.
    """
    b, c, h, w = depth.shape
    img_down = F.interpolate(img, (h, w), mode='bilinear', align_corners=True)
    img_down = torch.mean(img_down, dim=1, keepdim=True)
    
    # Compute gradients for image and depth
    grad_img = sobel_operator(img_down)
    grad_depth = sobel_operator(depth)
    
    # Weight depth gradients by image gradients
    weight = torch.exp(-grad_img)
    
    # Compute the edge-aware smoothness loss
    smoothness_loss = torch.mean(weight * grad_depth)
    
    return smoothness_loss