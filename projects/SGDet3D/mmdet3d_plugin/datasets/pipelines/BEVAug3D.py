import torch
import numpy as np
from PIL import Image
from mmdet.datasets.builder import PIPELINES
import mmcv
from copy import deepcopy
import numpy as np
from ...utils.visualization import draw_bev_pts_bboxes
from mmdet3d.core import show_multi_modality_result        
import cv2

@PIPELINES.register_module()
class GlobalRotScaleTransFlipAll:
    """Random resize, Crop and flip the image
    IT ACTUALLY DOING MODIFIED CAM EXTRINSICS PARAMETRS
    THUS UPDATE 'lidar2img' and 'lidar2cam'
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self,  bda_aug_conf, is_train=True):

        self.is_train = is_train
        self.bda_aug_conf = bda_aug_conf
        if not is_train: 
            bda_aug_conf = dict(
                rot_range=(0.00, 0.00),
                scale_ratio_range=(1.00, 1.00),
                translation_std=(0.0, 0.0, 0.0),
                flip_dx_ratio=0.0,
                flip_dy_ratio=0.0,
            ) 
        self.rot_range = bda_aug_conf['rot_range']
        self.scale_ratio_range = bda_aug_conf['scale_ratio_range']
        self.translation_std = bda_aug_conf['translation_std']
        self.flip_dx_ratio = bda_aug_conf['flip_dx_ratio']
        self.flip_dy_ratio = bda_aug_conf['flip_dy_ratio']
            
        # self.gt_img_visualizer = Det3DLocalVisualizer()
        # self.gt_bev_visualizer = Det3DLocalVisualizer()
    def __call__(self, results: dict) -> dict:
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """

        # random rotate
        if 'transformation_3d_flow' not in results:
            results['transformation_3d_flow'] = []
            
        # rotate
        self.rotate_bev_along_z(results)
        # translation
        self.trans_bbox_points(results)
        # random scale
        self.scale_xyz(results)
        # flip xy with random ratio
        self.flip_xy(results)
        
        # recording BEV-lidar augumentation
        results['transformation_3d_flow'].extend(['R', 'T', 'S', 'F'])
        lidar_augs = np.eye(4)
        lidar_augs[:3, :3] = results['pcd_flip_xy'][:3,:3] @ (results['pcd_rotation'][:3, :3] * results['pcd_scale_factor']) 
        lidar_augs[:3, 3] = results['pcd_flip_xy'][:3,:3] @ (results['pcd_trans'] * results['pcd_scale_factor'])
        if 'lidar_aug_matrix' not in results:
            results['lidar_aug_matrix'] = np.eye(4)
        results['lidar_aug_matrix'] = lidar_augs @ results['lidar_aug_matrix']
        results['bda_rot'] = results['lidar_aug_matrix']

        return results


    def rotate_bev_along_z(self, results):
        
        angle = np.random.uniform(*self.rot_range)
        if 'points' in results:
            results['points'].rotate(np.array(-angle)) # NOTE: angle or -angle?
        if "gt_bboxes_3d" in results:
            results["gt_bboxes_3d"].rotate(np.array(angle))
   
        results['pcd_rotation_angle'] = angle
    
        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))
   
        rot_mat_T = torch.tensor([[rot_cos, rot_sin, 0, 0], 
                        [-rot_sin, rot_cos, 0, 0], 
                        [0, 0, 1, 0], 
                        [0, 0, 0, 1]])
   
        results['pcd_rotation'] = rot_mat_T
        return

    def scale_xyz(self, results):
        
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        if 'points' in results:
            results['points'].scale(scale_ratio)
        if "gt_bboxes_3d" in results:
            results["gt_bboxes_3d"].scale(scale_ratio)
        results['pcd_scale_factor'] = scale_ratio
        
        return

    def trans_bbox_points(self, results):
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T
        if 'points' in results:
            results['points'].translate(trans_factor)
        if "gt_bboxes_3d" in results:
            results['gt_bboxes_3d'].translate(trans_factor)
        results['pcd_trans'] = torch.tensor(trans_factor).to(torch.float32)
    
            
    def flip_xy(self, results):
        mat = torch.tensor(
            [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]
            ]
        )
        if np.random.rand() < self.flip_dx_ratio:
            mat[0][0] = -1
            if 'points' in results:
                results['points'].flip(bev_direction='vertical')
            if "gt_bboxes_3d" in results:
                results["gt_bboxes_3d"].flip(bev_direction='vertical')
            results["pcd_vertical_flip"] = True
        if np.random.rand() < self.flip_dy_ratio:
            mat[1][1] = -1
            if 'points' in results:
                results['points'].flip(bev_direction='horizontal')
            if "gt_bboxes_3d" in results:
                results["gt_bboxes_3d"].flip(bev_direction='horizontal')
            results["pcd_horizontal_flip"] = True
            
        results['pcd_flip_xy'] = mat
        
    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(is_train={self.is_train},'
        repr_str += f'(rot_range={self.rot_range},'
        repr_str += f'(scale_ratio_range={self.scale_ratio_range},'
        repr_str += f'(translation_std={self.translation_std},'
        repr_str += f'(flip_dx_ratio={self.flip_dx_ratio},'
        repr_str += f'(flip_dy_ratio={self.flip_dy_ratio},'
        return repr_str