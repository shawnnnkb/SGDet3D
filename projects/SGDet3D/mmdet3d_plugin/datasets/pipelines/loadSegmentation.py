import os
import torch
import cv2
import numpy as np
from mmdet.datasets.builder import PIPELINES
from ....mmdet3d_plugin.utils.visualization import draw_bev_pts_bboxes, colorize
from PIL import Image
from mmdet3d.core.points import LiDARPoints

@PIPELINES.register_module()
class loadSegmentation:
    def __init__(self,
        data_root=None,
        dataset='kitti',
        seg_type='detectron2',
    ):
        self.data_root = data_root
        self.dataset = dataset
        self.seg_type = seg_type

        assert self.dataset in ['kitti', 'VoD', 'TJ4D']

    def __call__(self, results):
        
        segmt_pa_root = os.path.join(self.data_root, '..', 'segmentation')
        index_num = results['pts_filename'].split('/')[-1].split('.')[0]
  
        if self.dataset == 'VoD' and self.seg_type=='detectron2':
            # preparation of reading
            segmt_pa_path = os.path.join(segmt_pa_root, index_num + '.npy')
            segmt_pa = np.load(segmt_pa_path)
            # segmt_pa_path = os.path.join(segmt_pa_root, index_num + '.png')
            # segmt_pa = np.array(Image.open(segmt_pa_path), dtype=np.bool_)

        if self.dataset == 'TJ4D' and self.seg_type=='detectron2':
            segmt_pa_root = os.path.join(self.data_root, 'segmentation')
            segmt_pa_path = os.path.join(segmt_pa_root, index_num + '.npy')
            segmt_pa = np.load(segmt_pa_path)
            segmt_pa[(segmt_pa.shape[0]-160):, :] = 0 # check
            
        if self.dataset == 'kitti':
            pass
        
        # color_depth_map = colorize(depth_dc, vmin=0.0, vmax=80.0)
        # Image.fromarray(color_depth_map).save('color_depth.png')
        results['segmentation'] = segmt_pa
        
        return results