import os
import torch
import cv2
import numpy as np
from mmdet.datasets.builder import PIPELINES
from PIL import Image
from mmdet3d.core.points import LiDARPoints
import mmcv
from copy import deepcopy

@PIPELINES.register_module()
class CreateDepthFromRaDAR:
    def __init__(self, filter_min, filter_max):
        # filter_min_max
        self.filter_min = filter_min
        self.filter_max = filter_max
        
    def __call__(self, results):
        H, W = results['img_shape']
        img_metas = deepcopy(results)
        rots, trans, intrins, post_rots, post_trans = results['cam_aware'][:5]
        img_metas['cam2img'][:2, :3] = post_rots[:2, :2] @ img_metas['cam2img'][:2, :3]
        img_metas['cam2img'][:2, 2] = post_trans[:2] + img_metas['cam2img'][:2, 2]
        img_metas['lidar2img'] = img_metas['cam2img'] @ img_metas['lidar2cam']
        img_metas['lidar2img'] = img_metas['lidar2img'] @ np.linalg.inv(img_metas['lidar_aug_matrix'])
        lidar2img = img_metas['lidar2img']
        radar_depth = np.zeros((H, W))
        radar_point = results['points'].tensor.numpy()[:, :3]
        # Apply the lidar to image transformation
        pts_hom = np.concatenate((radar_point, np.ones((radar_point.shape[0], 1))), axis=1)  # Convert to homogeneous coordinates
        img_pts = np.matmul(lidar2img, pts_hom.T).T  # Apply transformation
        img_pts[:, :2] = img_pts[:, :2] / img_pts[:, 2:3]  # Convert back to Cartesian coordinates
        # Filter points that are within the image boundaries
        valid_mask = (img_pts[:, 0] >= 0) & (img_pts[:, 0] < W) & (img_pts[:, 1] >= 0) & (img_pts[:, 1] < H)
        valid_mask = valid_mask & (img_pts[:, 2]>self.filter_min) & (img_pts[:, 2]<self.filter_max) 
        img_pts = img_pts[valid_mask]
        # Extract the depth values and update the depth image
        depth_values = img_pts[:, 2]
        x_indices = np.floor(img_pts[:, 0]).astype(int)
        y_indices = np.floor(img_pts[:, 1]).astype(int)
        radar_depth[y_indices, x_indices] = depth_values
        results['radar_depth'] = radar_depth
        
        return results