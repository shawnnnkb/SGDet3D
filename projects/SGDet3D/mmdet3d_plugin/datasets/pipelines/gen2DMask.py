import os
import torch
import cv2
import numpy as np
from mmdet.datasets.builder import PIPELINES
from PIL import Image
from mmdet3d.core.points import LiDARPoints
import mmcv

@PIPELINES.register_module()
class gen2DMask:
    def __init__(self, use_seg=False, use_softlabel=False, is_train=True):
        # use preprocessing seg
        self.is_train = is_train
        self.use_seg = use_seg
        self.use_softlabel = use_softlabel
    def __call__(self, results):

        if not self.is_train:
            H, W = results['img_shape']
            bbox_Mask = np.zeros((H, W), dtype=np.bool_)
            results['bbox_Mask'] = bbox_Mask.astype(np.float32)
            return results
        
        H, W = results['img_shape']
        bbox_Mask = np.zeros((H, W), dtype=np.bool_)
        
        gt_bboxes = results['gt_bboxes'][results['gt_labels']!=-1]
        gt_bboxes = [gt_bboxes[i] for i in range(len(gt_bboxes))]

        # whether using softlabel
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        for bbox in gt_bboxes:
            x1, y1, x2, y2 = np.array(bbox).astype(np.int32)
            if not self.use_softlabel:
                bbox_Mask[y1:y2, x1:x2] = True
            else:
                y_center = (y1 + y2) / 2
                x_center = (x1 + x2) / 2
                bbox_height = y2 - y1 + 1
                bbox_width_ = x2 - x1 + 1
                sigma = min(bbox_height, bbox_width_) / 6
                gaussian_map = self.gaussian_2d(x_coords, y_coords, x_center, y_center, sigma)
                bbox_Mask = np.maximum(bbox_Mask, gaussian_map)
        # cv2.imwrite('bbox_Mask.png', 255.0*bbox_Mask)
        # cv2.imwrite('correspoding_img.png', results['img'])
        results['bbox_Mask'] = bbox_Mask.astype(np.float32)
        
        return results

    def gaussian_2d(self, x, y, x0, y0, sigma=1):
        return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))