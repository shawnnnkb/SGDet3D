import os
import torch
import cv2
import numpy as np
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class CreateDepthFromLiDAR:
    def __init__(self,
        data_root=None,
        dataset='kitti'
    ):
        self.data_root = data_root
        self.dataset = dataset
        assert self.dataset in ['kitti', 'VoD', 'TJ4D']

    def __call__(self, results):
        # rots, trans, intrins, post_rots, post_trans = results['cam_aware'][:5]
        # _, sensor2sensors, focal_length, baseline = results['cam_aware'][5:]
        imgs = results['img']
        img_h, img_w = imgs.shape[:2]
        
        if self.dataset == 'kitti':
            train_or_test = results['img_path'].split("/")[-3] # using velodyne_reduced or velodyne
            lidar_filename = os.path.join(self.data_root, train_or_test, "velodyne", results['lidar_points']['lidar_path'])
            lidar_points = np.fromfile(lidar_filename, dtype=np.float32).reshape(-1, 4)
            lidar_points = torch.from_numpy(lidar_points[:, :3]).float()
            rots, trans, intrins, post_rots, post_trans = results['cam_aware'][:5]
        if self.dataset == 'VoD':
            lidar_root = os.path.join(self.data_root.split("/")[0],self.data_root.split("/")[1], 'lidar')
            training_or_testing = results['pts_filename'].split('/')[3]
            index_num = results['pts_filename'].split('/')[5].split('.')[0]
            lidar_calib_path = os.path.join(lidar_root, training_or_testing, 'calib', index_num+'.txt')
            with open(lidar_calib_path, 'r') as f:
                lines = f.readlines()
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]]).reshape([3, 4]); P2 = self._extend_matrix(P2)
            rect = np.array([float(info) for info in lines[4].split(' ')[1:10]]).reshape([3, 3]); rect_4x4 = np.zeros([4, 4], dtype=rect.dtype); rect_4x4[3, 3] = 1.; rect_4x4[:3, :3] = rect; rect = rect_4x4
            Trv2c = np.array([float(info) for info in lines[5].split(' ')[1:13]]).reshape([3, 4]); Trv2c = self._extend_matrix(Trv2c)
            lidar2cam = rect @ Trv2c
            lidar2cam = torch.Tensor(lidar2cam)
            cam2lidar = lidar2cam.inverse()
            rots = cam2lidar[:3, :3] # rot @ cam_3d = lidar_3d, here cam_3d is Column vector
            trans = cam2lidar[:3, 3] # cam_3d + tran = lidar_3d
            intrins, post_rots, post_trans = results['cam_aware'][2:5]
            lidar_filename = lidar_calib_path.replace('calib', 'velodyne').replace('txt','bin')
            lidar_points = np.fromfile(lidar_filename, dtype=np.float32).reshape(-1, 4)
            lidar_points = torch.from_numpy(lidar_points[:, :3]).float()
        if self.dataset == 'TJ4D':
            # NO LIDAR provided
            lidar_root = self.data_root
            training_or_testing = results['pts_filename'].split('/')[2]
            index_num = results['pts_filename'].split('/')[-1].split('.')[0]
            lidar_calib_path = os.path.join(lidar_root, training_or_testing, 'calib', index_num+'.txt')
            with open(lidar_calib_path, 'r') as f:
                lines = f.readlines()
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]]).reshape([3, 4]); P2 = self._extend_matrix(P2)
            rect = np.array([float(info) for info in lines[4].split(' ')[1:10]]).reshape([3, 3]); rect_4x4 = np.zeros([4, 4], dtype=rect.dtype); rect_4x4[3, 3] = 1.; rect_4x4[:3, :3] = rect; rect = rect_4x4
            Trv2c = np.array([float(info) for info in lines[5].split(' ')[1:13]]).reshape([3, 4]); Trv2c = self._extend_matrix(Trv2c)
            lidar2cam = rect @ Trv2c
            lidar2cam = torch.Tensor(lidar2cam)
            cam2lidar = lidar2cam.inverse()
            rots = cam2lidar[:3, :3] # rot @ cam_3d = lidar_3d, here cam_3d is Column vector
            trans = cam2lidar[:3, 3] # cam_3d + tran = lidar_3d
            intrins, post_rots, post_trans = results['cam_aware'][2:5]
            lidar_filename = lidar_calib_path.replace('calib', 'velodyne').replace('txt','bin')
            lidar_points = np.fromfile(lidar_filename, dtype=np.float32).reshape(-1, 4)
            lidar_points = torch.from_numpy(lidar_points[:, :3]).float()
        
        # [num_point, num_img, 3] in format [u, v, d]
        projected_points = self.project_points(lidar_points, rots, trans, intrins, post_rots, post_trans)

        valid_mask = (projected_points[..., 0] >= 0) & \
                    (projected_points[..., 1] >= 0) & \
                    (projected_points[..., 0] <= img_w - 1) & \
                    (projected_points[..., 1] <= img_h - 1) & \
                    (projected_points[..., 2] > 0) & \
                    (projected_points[..., 2] <= 80)
                    
        gt_depths = torch.zeros((img_h, img_w))
        valid_points = projected_points[valid_mask]
        # sort
        depth_order = torch.argsort(valid_points[:, 2], descending=True)
        valid_points = valid_points[depth_order]
        # fill in
        gt_depths[valid_points[:, 1].round().long(), valid_points[:, 0].round().long()] = valid_points[:, 2]
        results['gt_depths'] = gt_depths
        
        return results

    def project_points(self, points, rots, trans, intrins, post_rots, post_trans):
        '''
            rots, trans: original camera2lidar extrinsics
            post_rots, post_trans: image aug caused, same as change cam2img
        '''
        
        # preprocessing initialization
        points = points.view(-1, 1, 3) # N, 1, 3
        points = points - trans.view(1, -1, 3) # N, v, 3
        inv_rots = rots.inverse().view(-1, 3, 3).unsqueeze(0) # 1, v, 3, 3
        points = (inv_rots @ points.unsqueeze(-1)) # 1, v, 3, 3 @ N, v, 3, 1 = N, v, 3, 1
        
        # from lidar to camera
        points = torch.cat((points, torch.ones((points.shape[0], points.shape[1], 1, 1))), dim=2) # N, v, 4, 1
        points = (intrins.view(-1, 4, 4).unsqueeze(0) @ points).squeeze(-1) # N, v, 4
        
        # post processing for depth
        points_d = points[..., 2:3] # N, v, 1
        points_uv = points[..., :2] / points_d # N, v, 2
        
        # from raw pixel to transformed pixel
        points_uv = post_rots[:2, :2].view(-1, 2, 2).unsqueeze(0) @ points_uv.unsqueeze(-1)
        points_uv = points_uv.squeeze(-1) + post_trans.view(-1, 3)[:, :2].unsqueeze(0)
        points_uvd = torch.cat((points_uv, points_d), dim=2)
        
        return points_uvd
    
    def _extend_matrix(self, mat):
        mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
        return mat