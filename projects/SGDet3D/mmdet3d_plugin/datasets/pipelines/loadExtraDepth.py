import os
import torch
import cv2
import numpy as np
from mmdet.datasets.builder import PIPELINES
from ....mmdet3d_plugin.utils.visualization import draw_bev_pts_bboxes, colorize
from PIL import Image
from mmdet3d.core.points import LiDARPoints

@PIPELINES.register_module()
class loadExtraDepth:
    def __init__(self,
        data_root=None,
        dataset='kitti',
        depth_type='depth_anything_crop400',
    ):
        self.data_root = data_root
        self.dataset = dataset
        self.depth_type = depth_type

        assert self.dataset in ['kitti', 'VoD', 'TJ4D']

    def __call__(self, results):
        
        depth_dc_root = os.path.join(self.data_root, '..', self.depth_type)
        index_num = results['pts_filename'].split('/')[-1].split('.')[0]
        if self.dataset == 'VoD' and self.depth_type=='depth_anything_upfromds2':
            # preparation of reading
            depth_dc_path = os.path.join(depth_dc_root, index_num + '.npy')
            depth_dc = np.load(depth_dc_path)
            depth_dc [depth_dc <= 0] = 0
            
        if self.dataset == 'VoD' and self.depth_type=='depth_anything_crop400':
            # preparation of reading
            depth_dc_path = os.path.join(depth_dc_root, index_num + '.npy')
            depth_dc = np.load(depth_dc_path)
            depth_dc [depth_dc <= 0] = 0
            
        if self.dataset == 'VoD' and self.depth_type=='NLSPN': 
            # preparation of reading
            depth_dc_path = os.path.join(depth_dc_root, index_num + '.npy')
            depth_dc = np.load(depth_dc_path)
            depth_dc [depth_dc <= 0] = 0
            # NOTE: here is the mistakes of ZFY, just downsample ratio 2, min=0.0, max=70.0
            H, W = depth_dc.shape # NOTE: up sample method should be thought
            depth_dc = cv2.resize(depth_dc, (W*2, H*2), interpolation=cv2.INTER_LINEAR) # bilinear
            
        if self.dataset == 'TJ4D':
            pass
        if self.dataset == 'kitti':
            pass
        
        # color_depth_map = colorize(depth_dc, vmin=0.0, vmax=80.0)
        # Image.fromarray(color_depth_map).save('color_depth.png')
        results['depth_comple'] = depth_dc
        
        return results