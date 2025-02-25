import mmcv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Any, Dict
from mmdet.datasets.builder import PIPELINES
import mmcv
from copy import deepcopy    
from ....mmdet3d_plugin.utils.visualization import draw_bev_pts_bboxes, colorize
import cv2   
from ..structures.bbox import HorizontalBoxes

@PIPELINES.register_module()
class ImageAug3D():
    """ modified from YZ and BEVFusion

    Args:
        final_dim: target dimensions of the final image, specified as (height, width).
        resize_lim: Color type of the file. Defaults to 'unchanged'.
        bot_pct_lim: the range of cropping from the bottom of the image.
        top_pct_lim: the range of cropping from the top of the image.
        rot_lim: Indicates the range limits for rotation angle
        rand_flip:  A boolean value determining whether to perform random flipping (left-right) or not.
        is_train: A boolean value indicating whether the model is in training mode or not.
    """
    def __init__(self, data_aug_conf, is_train):
        super().__init__()
        
        if not is_train:
            data_aug_conf = {
            'resize_lim': (1.00, 1.00),
            'final_dim': data_aug_conf['final_dim_test'],
            'bot_pct_lim': (0.0, 0.0),
            'top_pct_lim': (0.0, 0.0),
            'rot_lim': (0.0, 0.0),
            'rand_flip': False}
            
        self.data_aug_conf = data_aug_conf
        self.final_dim = data_aug_conf['final_dim']
        self.resize_lim = data_aug_conf['resize_lim']
        self.bot_pct_lim = data_aug_conf['bot_pct_lim']
        self.top_pct_lim = data_aug_conf['top_pct_lim']
        self.rand_flip = data_aug_conf['rand_flip']
        self.rot_lim = data_aug_conf['rot_lim']
        self.bbox_clip_border = True # for 2d bbox
        self.is_train = is_train

    def sample_augmentation(self, results):
        H, W = results['ori_shape'][0], results['ori_shape'][1]
        fH, fW = self.final_dim
        results['img_shape'] = (fH, fW)
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            # crop_h = int((1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_h = int(np.random.uniform(*self.top_pct_lim) * newH)
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            # resize = np.mean(self.resize_lim)
            resize = np.min([fH / H, fW / W])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            # if not (fH==H and fW==W):
            #     print('WARNING: testing final_dim_test not equal to ori_shape, resize_dims:(%d,%d), final_dim:(%d,%d),  ori_shape:(%d,%d)'%(newH, newW, fH, fW, H, W))
            # crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_h = int(np.random.uniform(*self.top_pct_lim) * newH)
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.rand_flip; assert flip == False, 'testing time flip should be False'
            rotate = np.mean(self.rot_lim); assert rotate == 0, 'testing time rotate should be 0'
        return resize, resize_dims, crop, flip, rotate

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims, crop, flip, rotate):
        # adjust image
        img = Image.fromarray(img.astype('uint8'), mode='RGB')
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        
        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ])
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # data['cam2img'] @ data['lidar2cam']==data['lidar2img']
        img = data['img'] # [:,:,(2,1,0)] # BGR->RGB
        focal_length = data['focal_length']
        baseline = data['baseline']
        
        # perform image-view augmentation
        # initilize post_rot and post_trans
        # here, actually is pixel space tran & rot
        post_rot = torch.eye(2)
        post_trans = torch.zeros(2)
        resize, resize_dims, crop, flip, rotate = self.sample_augmentation(data)
        img, post_rot2, post_tran2 = self.img_transform(
            img, 
            post_rot, 
            post_trans, 
            resize=resize, 
            resize_dims=resize_dims, 
            crop=crop, 
            flip=flip, 
            rotate=rotate
        ) # here retrun post_rot2, post_tran2
        # was caused by resize-crop-flip-rotate aug
        if data.get('gt_bboxes', None) is not None:
            data = self.gt_bboxes_transform(
                data,
                img=img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate
            )
        
        if "depths" in data.keys():
            depths = data['depths']
            centers_2d = data['centers_2d']
            input = np.concatenate([centers_2d, depths.reshape(-1,1)], axis=-1)
            # modify centers_2d and draw a depth map (H, W, 1) where centers_2d is not 0
            depth_map, centers_2d = self._depth_transform(
                input,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            data['centers_2d'] = centers_2d
            data['depth_map'] = depth_map
        
        if 'segmentation' in data.keys():
            
            map = data['segmentation']
            map = Image.fromarray(map)
            map = map.resize(resize_dims)
            map = map.crop(crop)
            if flip: map = map.transpose(method=Image.FLIP_LEFT_RIGHT)
            map = map.rotate(rotate)
            segmentation = np.array(map).astype(np.bool_)
            data['segmentation'] = segmentation
            
        if 'depth_comple' in data.keys():
            map = data['depth_comple']
            map = Image.fromarray(map)
            map = map.resize(resize_dims)
            map = map.crop(crop)
            if flip: map = map.transpose(method=Image.FLIP_LEFT_RIGHT)
            map = map.rotate(rotate)
            depth_comple = np.array(map).astype(np.float32)
            # color_depth_map = colorize(depth_comple, vmin=0.0, vmax=80.0)
            # Image.fromarray(color_depth_map).save('color_depth.png')
            data['depth_comple'] = depth_comple

        img = np.array(img).astype(np.float32)
        data['img'] = img

        # BEVFusion like recording transform matrix caused by img aug
        transform = torch.eye(4)
        transform[:2, :2] = post_rot2
        transform[:2, 3] = post_tran2
        data['img_aug_matrix'] = transform.numpy()
        
        # for convenience, make augmentation matrices 3x3
        # here, actually is pixel space tran & rot
        post_rot = torch.eye(3)
        post_tran = torch.zeros(3)
        post_rot[:2, :2] = post_rot2 # post_rot @ uvd = uvd', here uvd is Column vector
        post_tran[:2] = post_tran2 # post_tran + uvd = uvd'

        # intrins
        intrin = torch.Tensor(data['cam2img'])

        # extrins
        lidar2cam = torch.Tensor(data['lidar2cam'])
        cam2lidar = lidar2cam.inverse()
        rot = cam2lidar[:3, :3] # rot @ cam_3d = lidar_3d, here cam_3d is Column vector
        tran = cam2lidar[:3, 3] # cam_3d + tran = lidar_3d

        # output
        depth = torch.zeros(1)
        cam_aware = [rot, tran, intrin, post_rot, post_tran, depth, cam2lidar]
        cam_aware.append(torch.tensor(focal_length, dtype=torch.float32))
        cam_aware.append(torch.tensor(baseline, dtype=torch.float32))
        data['cam_aware'] = cam_aware

        return data
    def gt_bboxes_transform(self, results, img, resize, resize_dims, crop, flip, rotate):
        results['gt_bboxes'] = HorizontalBoxes(results['gt_bboxes'], in_mode='xyxy')
        if resize: results['gt_bboxes'].tensor = results['gt_bboxes'].tensor * resize
        if crop: 
            offset_w, offset_h = crop[0], crop[1] 
            bboxes = results['gt_bboxes']
            bboxes.translate_([-offset_w, -offset_h])
            if self.bbox_clip_border: bboxes.clip_(self.final_dim)
            valid_inds = bboxes.is_inside(self.final_dim).numpy()
            results['gt_bboxes'] = bboxes[valid_inds]
            results['gt_labels'] = results['gt_labels'][valid_inds]
            results['gt_bboxes_3d'] = results['gt_bboxes_3d'][valid_inds]
            results['gt_labels_3d'] = results['gt_labels_3d'][valid_inds]
            if 'centers_2d' in results: results['centers_2d'] = results['centers_2d'][valid_inds]
            if 'depths' in results: results['depths'] = results['depths'][valid_inds]
        if flip: 
            results['gt_bboxes'].flip_(self.final_dim, 'horizontal')
        if rotate: 
            rotn_center = (self.final_dim[1] / 2.0, self.final_dim[0] / 2.0)
            results['gt_bboxes'].rotate_(rotn_center, -rotate)
        
        results['gt_bboxes'] = results['gt_bboxes'].tensor
        return results
    def _depth_transform(self, cam_depth, resize, resize_dims, crop, flip, rotate):
        """
        Input:
            cam_depth: Nx3, 3: x,y,d
            resize: a float value
            resize_dims: self.ida_aug_conf["final_dim"] -> [H, W]
            crop: x1, y1, x2, y2
            flip: bool value
            rotate: an angle
        Output:
            cam_depth: [h/down_ratio, w/down_ratio, d]
        """

        H, W = resize_dims
        cam_depth[:, :2] = cam_depth[:, :2] * resize
        cam_depth[:, 0] -= crop[0]
        cam_depth[:, 1] -= crop[1]
        if flip:
            cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

        cam_depth[:, 0] -= W / 2.0
        cam_depth[:, 1] -= H / 2.0

        h = rotate / 180 * np.pi
        rot_matrix = [
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ]
        cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

        cam_depth[:, 0] += W / 2.0
        cam_depth[:, 1] += H / 2.0

        depth_coords = cam_depth[:, :2].astype(np.int16)

        depth_map = np.zeros((H, W, 1))
        valid_mask = (
            (depth_coords[:, 1] < resize_dims[0])
            & (depth_coords[:, 0] < resize_dims[1])
            & (depth_coords[:, 1] >= 0)
            & (depth_coords[:, 0] >= 0)
        )
        depth_map[depth_coords[valid_mask, 1], depth_coords[valid_mask, 0], :] = cam_depth[valid_mask, 2:3]
        depth_map = depth_map.astype(np.float32)
        
        return depth_map, depth_coords
