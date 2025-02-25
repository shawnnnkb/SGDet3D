import torch, copy, time, os, mmcv
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from shapely.geometry import Polygon, box, Point
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mmcv.runner.dist_utils import master_only
from mmdet.models import DETECTORS
from mmdet.models.backbones.resnet import BasicBlock
from mmdet3d.models import builder
from mmdet3d.models.builder import FUSION_LAYERS, NECKS
from mmdet3d.models.detectors import MVXFasterRCNN
from mmdet3d.core import bbox3d2result, show_multi_modality_result
from ...datasets.structures.bbox import HorizontalBoxes
from ...utils.visualization import draw_bev_pts_bboxes, draw_paper_bboxes
from ...utils.visualization import custom_draw_lidar_bbox3d_on_img


@DETECTORS.register_module()
class SGDet3D(MVXFasterRCNN):
    """Multi-modality BEVFusion using Faster R-CNN."""

    def __init__(self, 
                bev_h_=160,
                bev_w_=160,
                img_channels=256, 
                rad_channels=384,
                num_classes=3,
                num_in_height=8,
                use_depth_supervision=True,
                use_props_supervision=True,
                use_box3d_supervision=True,
                use_msk2d_supervision=True,
                use_backward_projection=False,
                use_sa_radarnet=False,
                use_grid_mask=False,
                freeze_images=True,
                freeze_depths=False,
                freeze_radars=False,
                camera_stream='LSS',
                point_cloud_range=None,
                grid_config=None,
                img_norm_cfg=None,
                # ablation settings
                backward_ablation=None,
                focusradardepth_ablation=None,
                painting_ablation=None,
                # model config
                depth_net=None,
                rangeview_foreground=None,
                img_view_transformer=None,
                RCFusion=None,
                proposal_layer=None,
                backward_projection=None,
                meta_info=None,
                **kwargs):
        self.pts_bbox_head = kwargs.pop('pts_bbox_head')
        self.pts_dim = kwargs['pts_voxel_encoder']['in_channels']
        self.pts_config = {}
        self.pts_config.update(
            pts_voxel_encoder = kwargs['pts_voxel_encoder'],
            pts_middle_encoder = kwargs['pts_middle_encoder'],
            pts_backbone = kwargs['pts_backbone'],
            pts_neck = kwargs['pts_neck'])
        super(SGDet3D, self).__init__(**kwargs)

        # hyper-parameter settings
        self.bev_h_ = bev_h_
        self.bev_w_ = bev_w_
        self.img_channels = img_channels
        self.rad_channels = rad_channels
        self.num_classes = num_classes
        self.num_in_height = num_in_height
        self.freeze_images = freeze_images
        self.freeze_depths = freeze_depths
        self.freeze_radars = freeze_radars
        self.lift_method = camera_stream   
        self.point_cloud_range = point_cloud_range
        self.grid_config = grid_config
        self.img_norm_cfg = img_norm_cfg
        self.use_grid_mask = use_grid_mask
        self.use_depth_supervision = use_depth_supervision
        self.use_props_supervision = use_props_supervision
        self.use_box3d_supervision = use_box3d_supervision
        self.use_msk2d_supervision = use_msk2d_supervision
        self.use_backward_projection = use_backward_projection

        self.use_sa_radarnet = use_sa_radarnet
        self.use_radar_depth = depth_net['use_radar_depth']
        self.use_extra_depth = depth_net['use_extra_depth']
        self.backward_ablation=backward_ablation
        self.focusradardepth_ablation=focusradardepth_ablation
        self.painting_ablation=painting_ablation
        self.RCFusion = RCFusion
        self.meta_info = meta_info
        self.figures_path = meta_info['figures_path']
        self.poject_name = meta_info['poject_name']
        if 'vod' in self.poject_name.lower(): self.dataset_type = 'VoD'
        if 'tj4d' in self.poject_name.lower(): self.dataset_type = 'TJ4D'

        # other papa for convenience
        self.xbound = self.grid_config['xbound']
        self.ybound = self.grid_config['ybound']
        self.zbound = self.grid_config['zbound']
        self.bev_grid_shape = [bev_h_, bev_w_]
        self.bev_cell_size = [(self.xbound[1]-self.xbound[0])/bev_h_, (self.ybound[1]-self.ybound[0])/bev_w_]
        self.voxel_size = [self.grid_config['xbound'][2], self.grid_config['ybound'][2], self.grid_config['zbound'][2]]
        self.backward_use_pv_logits = self.backward_ablation['pv_logits']
        self.backward_use_depth_prob = self.backward_ablation['depth_prob']
        self.painting_use_pv_logits = self.painting_ablation['pv_logits']
        self.painting_use_depth_prob = self.painting_ablation['depth_prob']
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        self.xlim, self.ylim = [x_min, x_max], [y_min, y_max]
        
        # vanilla model and loss settings
        if self.lift_method == 'OFT': pass
        if self.lift_method == 'LSS':
            depth_net.update(figures_path=self.figures_path)
            self.depth_net = FUSION_LAYERS.build(depth_net) if depth_net else None
            img_view_transformer.update(num_in_height=self.num_in_height)
            self.img_view_transformer = FUSION_LAYERS.build(img_view_transformer)
            self.downsample = self.depth_net.downsample
        self.rangeview_foreground = NECKS.build(rangeview_foreground) if (rangeview_foreground and use_msk2d_supervision) else None
        self.cross_attention = FUSION_LAYERS.build(RCFusion)
        self.proposal_layer_former = FUSION_LAYERS.build(proposal_layer) if (proposal_layer and use_props_supervision) else None
        if self.use_backward_projection: # bev latter means after RCFusion (use paint BEV fusion or backward projection)
            self.proposal_layer_latter = FUSION_LAYERS.build(proposal_layer) if (proposal_layer and use_props_supervision) else None
        else: self.proposal_layer_latter = None
        self.backward_projection = FUSION_LAYERS.build(backward_projection) if (backward_projection and use_backward_projection) else None
        if self.pts_bbox_head and use_box3d_supervision:
            pts_train_cfg = self.train_cfg.pts if self.train_cfg else None
            self.pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = self.test_cfg.pts if self.test_cfg else None
            self.pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head_dict = self.pts_bbox_head
            self.pts_bbox_head = builder.build_head(self.pts_bbox_head)
    
        # init weights and freeze if needed
        self.init_flexible_modules()
        self.init_weights()
        if self.freeze_images: self.freeze_img_model()
        if self.freeze_radars: self.freeze_pts_model()
        self.record_fps = {'num': 0, 'time':0}
        self.init_visulization()
    
    # initialize modules
    
    def init_flexible_modules(self):
        if self.use_sa_radarnet:
            self.voxelpainting_points, self.voxel_coords = self.generate_pillar_ref_points(self.num_in_height)
            in_channels = self.num_in_height * self.img_channels
            self.adaptive_collapse_conv = nn.Sequential(
                nn.Conv2d(in_channels, self.rad_channels, kernel_size=1),
                BasicBlock(self.rad_channels, self.rad_channels))
            self.PaintBEVFusion = nn.Sequential(
                BasicBlock(self.rad_channels*2, self.rad_channels, downsample=nn.Conv2d(self.rad_channels*2, self.rad_channels, 3, 1, 1)),
                BasicBlock(self.rad_channels, self.rad_channels))
    def init_visulization(self):
        self.SAVE_INTERVALS = 250 # 500
        self.vis_time_box3d = 0
        self.vis_time_bev2d = 0
        self.vis_time_bevnd = 0
        self.vis_time_range = 0
        self.vis_time_point = 0
        self.mean=np.array(self.img_norm_cfg['mean'])
        self.std=np.array(self.img_norm_cfg['std'])
        self.figures_path_det3d_test = os.path.join(self.figures_path, 'test', 'det3d')
        self.figures_path_bev2d_test = os.path.join(self.figures_path, 'test', 'bev_mask')
        self.figures_path_bevnd_test = os.path.join(self.figures_path, 'test', 'bev_feats')
        self.figures_path_range_test = os.path.join(self.figures_path, 'test', 'range')
        self.figures_path_point_test = os.path.join(self.figures_path, 'test', 'point')
        self.figures_path_det3d_train = os.path.join(self.figures_path, 'train', 'det3d')
        self.figures_path_bev2d_train = os.path.join(self.figures_path, 'train', 'bev_mask')
        self.figures_path_bevnd_train = os.path.join(self.figures_path, 'train', 'bev_feats')
        self.figures_path_range_train = os.path.join(self.figures_path, 'train', 'range')
        self.figures_path_point_train = os.path.join(self.figures_path, 'train', 'point')
        os.makedirs(self.figures_path_det3d_test, exist_ok=True)
        os.makedirs(self.figures_path_bev2d_test, exist_ok=True)
        os.makedirs(self.figures_path_bevnd_test, exist_ok=True)
        os.makedirs(self.figures_path_range_test, exist_ok=True)
        os.makedirs(self.figures_path_point_test, exist_ok=True)
        os.makedirs(self.figures_path_det3d_train, exist_ok=True)
        os.makedirs(self.figures_path_bev2d_train, exist_ok=True)
        os.makedirs(self.figures_path_bevnd_train, exist_ok=True)
        os.makedirs(self.figures_path_range_train, exist_ok=True)
        os.makedirs(self.figures_path_point_train, exist_ok=True)
            
    # model parameter freezing or not 
    
    def freeze_img_model(self):
        """freeze image backbone and neck for fusion"""
        if self.with_img_backbone:
            for param in self.img_backbone.parameters():
                param.requires_grad = False
        if self.with_img_neck:
            for param in self.img_neck.parameters():
                param.requires_grad = False
        if self.lift_method == 'LSS' and self.freeze_depths:
            for param in self.depth_net.parameters():
                param.requires_grad = False
    
    def freeze_pts_model(self):
        """freeze radar backbone and neck for pretrain"""
        if self.pts_voxel_encoder:
            for param in self.pts_voxel_encoder.parameters():
                param.requires_grad = False
        if self.pts_middle_encoder:
            for param in self.pts_middle_encoder.parameters():
                param.requires_grad = False
        if self.pts_backbone:
            for param in self.pts_backbone.parameters():
                param.requires_grad = False
        if self.pts_neck is not None:
            for param in self.pts_neck.parameters():
                param.requires_grad = False
       
    # feature pre-extraction
    
    def generate_pillar_ref_points(self, num_in_height):
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        voxel_x, voxel_y, voxel_z = self.voxel_size
        
        # Calculate the center points for the grid
        x_centers = torch.arange(x_min + voxel_x / 2, x_max, voxel_x)
        y_centers = torch.arange(y_min + voxel_y / 2, y_max, voxel_y)
        z_step = (z_max - z_min) / num_in_height
        z_centers = torch.arange(z_min + z_step / 2, z_max, z_step)
        assert x_centers.shape[0] == self.bev_h_
        assert y_centers.shape[0] == self.bev_w_

        # Create a mesh grid for x, y, z
        xv, yv, zv = torch.meshgrid(x_centers, y_centers, z_centers)

        # Stack the grid coordinates
        ref_points = torch.stack((xv, yv, zv), dim=-1) # shape: (H, W, Z, 3)
        
        # indices
        hv, wv, zv = torch.meshgrid(torch.arange(self.bev_h_), torch.arange(self.bev_w_), torch.arange(num_in_height))
        idx = torch.arange(self.bev_h_ * self.bev_w_ * num_in_height)
        voxel_coords = torch.cat([hv.reshape(-1, 1), wv.reshape(-1, 1), zv.reshape(-1, 1), idx.reshape(-1, 1)], dim=-1)
        return ref_points, voxel_coords
    
    def extract_pts_feat(self, pts, img_metas):
        """Extract features of raw points."""
        if not self.with_pts_backbone: return None
        
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,)
        batch_size = coors[-1, 0].item() + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
                
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats
    
    # NOTE: core model here, processing multi-modality feats
    
    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        
        # preparation of camera-geo-aware input
        if img.dim() == 3 and img.size(0)== 3: img = img.unsqueeze(0)
        B, C, H, W = img.shape
        if not isinstance(points, list): points = [points]
        img_metas, gt_bboxes_3d, gt_labels_3d, gt_bboxes_2d, gt_labels_2d, depth_comple, bbox_Mask, segmentation, radar_depth, cam_aware, \
            img_aug_matrix, lidar_aug_matrix, bda_rot, gt_depths, gt_bev_mask, final_lidar2img = self.preprocessing_information(img_metas, img.device)
        img_inputs = [img, cam_aware[0], cam_aware[1], cam_aware[2], cam_aware[3], cam_aware[4], bda_rot]
        img, rots, trans, intrins, post_rots, post_trans, bda = img_inputs[0:7]
        matrix = torch.eye(4).to(final_lidar2img.device)
        matrix[0, 0] = matrix[0, 0] / self.downsample
        matrix[1, 1] = matrix[1, 1] / self.downsample
        projection = matrix @ final_lidar2img

        # 1. pre-extract img features of raw image 
        start_time = time.time()
        img_feats = self.extract_img_feat(img, img_metas)
        end_time = time.time()
        step1_time = end_time - start_time

        # 2. GDC: geometric depth completion, lift img context to bev feats
        if self.lift_method == 'LSS':
            mlp_input = self.depth_net.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)
            geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]
            view_trans_inputs = [rots, trans, intrins, post_rots, post_trans, bda]
            cam_params_list = [[rots[i:i+1], trans[i:i+1], intrins[i:i+1], post_rots[i:i+1], post_trans[i:i+1], bda[i:i+1]] for i in range(img.shape[0])]
            index = self.downsample // 4 - 1
            h, w = img.shape[2] // self.downsample, img.shape[3] // self.downsample
            align_feats = [F.interpolate(feat, (h, w), mode='bilinear', align_corners=True) for feat in img_feats]
            align_feats = torch.cat(align_feats, dim=1)
            start_time = time.time()
            context, depth = self.depth_net([align_feats] + geo_inputs, radar_depth, depth_comple, img_metas)
            end_time = time.time()
            img_bev_feats = self.img_view_transformer(context, depth, view_trans_inputs)
            img_bev_feats = img_bev_feats.mean(-1) # B, C, bev_h_, bev_w_
            img_bev_feats = img_bev_feats.permute(0, 1, 3, 2).contiguous()
            cam_depth_range = self.grid_config['dbound']
            raw_depth = torch.arange(cam_depth_range[0], cam_depth_range[1], cam_depth_range[2]).to(depth.device)
            precise_depth = torch.sum(raw_depth.view(1,-1,1,1)*depth, dim=1).unsqueeze(1)
        else: context, depth, precise_depth = None, None, None
        if self.lift_method == 'OFT': pass
        if self.lift_method == None: img_bev_feats = pts_bev_feats
        step2_time = end_time - start_time
        
        # 3. perspective view foreground segmentation
        if self.rangeview_foreground is not None and self.use_msk2d_supervision and context is not None:
            start_time = time.time()
            rangeview_logit = self.rangeview_foreground(context.squeeze(1)) 
            end_time = time.time()
            rangeview_logit_sigmoid = rangeview_logit.sigmoid()
            focus_weight = 1.0*precise_depth/self.point_cloud_range[3]
            mask_reweighted = (1-focus_weight)*rangeview_logit_sigmoid+focus_weight*torch.ones_like(rangeview_logit_sigmoid).to(context.device)
            min_vals = torch.min(mask_reweighted.reshape(B, -1), dim=1)[0].reshape(B, 1, 1, 1)
            max_vals = torch.max(mask_reweighted.reshape(B, -1), dim=1)[0].reshape(B, 1, 1, 1)
            mask_reweighted = (mask_reweighted-min_vals)/(max_vals-min_vals)
            rangeview_logit_sigmoid = mask_reweighted
        else: rangeview_logit = None
        step3_time = end_time - start_time
        
        # 4. SRP: use painting for point decorated
        if self.use_sa_radarnet:
            radar_decorate_img_feats = context.squeeze(1)
            pts_bev_feats = self.extract_pts_feat(points, img_metas)[0]
            h, w, z = self.voxelpainting_points.shape[:3]
            points_vis = [torch.cat([points[i], torch.zeros((points[i].shape[0], self.img_channels), device=context.device)], dim=-1) for i in range(B)]
            input_points = [self.voxelpainting_points.reshape(-1, 3).to(context.device) for _ in range(B)]
            start_time = time.time()
            all_decorated_points = self.voxelpainting_depth_aware(radar_decorate_img_feats, rangeview_logit.sigmoid(), depth, input_points, projection)
            end_time = time.time()
            feature_sum = [torch.sum(all_decorated_points[i][:, 5:], dim=1) for i in range(B)]
            topk_indice = [torch.topk(feature_sum[i], 200)[1] for i in range(B)]
            decorated_points_vis = [all_decorated_points[i][topk_indice[i]] for i in range(B)]
            self.draw_pts_completion(img_metas, decorated_points_vis, points_vis, gt_bboxes_3d, gt_labels_3d, plot_mode='context_voxelpainting', points_type='voxelpainting') # distance RCS v_r_compensated context
            all_decorated_points = [all_decorated_points[i][:, self.pts_dim:] for i in range(B)]
            paint_bev = [x.view(1, h, w, z, -1) for x in all_decorated_points]
            paint_bev = torch.cat(paint_bev, dim=0) # B H W Z C
            paint_bev = paint_bev.view(B, h, w, -1).permute(0, 3, 2, 1).contiguous() # B C*Z H W
            paint_bev = self.adaptive_collapse_conv(paint_bev)
            pts_bev_feats_origin = pts_bev_feats
            if self.dataset_type == 'VoD': paint_bev = 0.01*paint_bev # original implementation for VoD
            pts_bev_feats = self.PaintBEVFusion(torch.cat([paint_bev, pts_bev_feats], dim=1))
            end_time = time.time()
        else:  # without any painting
            pts_bev_feats = self.extract_pts_feat(points, img_metas)[0]
            paint_bev = torch.zeros_like(pts_bev_feats).to(context.device)
            pts_bev_feats_origin = pts_bev_feats
        step4_time = end_time - start_time
            
        # 5. BEV fusion radar and image feats
        start_time = time.time()
        bev_feats = self.cross_attention(img_bev_feats, pts_bev_feats)
        end_time = time.time()
        bev_feats = bev_feats.permute(0, 1, 3, 2).contiguous()
        assert bev_feats.shape[2] == self.bev_h_
        assert bev_feats.shape[3] == self.bev_w_
        step5_time = end_time - start_time

        # auxiliry BEV segementation loss
        if self.proposal_layer_former is not None and self.use_props_supervision:
            bev_mask_logit_former = self.proposal_layer_former(bev_feats)
        else: bev_mask_logit_former = None
        
        # 6. LACA backward projection, refine bev_feats further
        if self.backward_projection is not None and self.lift_method == 'LSS':
            bev_mask = torch.ones((B, 1, self.bev_h_, self.bev_w_), dtype=torch.bool).to(img.device)
            search_img_feats = img_feats[index]
            search_img_feats = search_img_feats*rangeview_logit_sigmoid if self.backward_use_pv_logits else search_img_feats
            search_img_feats = search_img_feats.unsqueeze(1)
            depth_dist = torch.full(depth.shape, 1.0 / depth.shape[1]).to(context.device) if not self.backward_use_depth_prob else depth
            start_time = time.time()
            bev_feats_refined = self.backward_projection(
                imgs = img,
                mlvl_feats = [search_img_feats],
                proposal = bev_mask,
                cam_params=cam_params_list,
                lss_bev=bev_feats,
                img_metas=img_metas,
                mlvl_dpt_dists=[depth_dist.unsqueeze(1)],
                backward_bev_mask_logit=torch.zeros_like(bev_feats).to(context.device))
            end_time = time.time()
            bev_feats_refined = bev_feats_refined.permute(0, 2, 1).view(B, self.img_channels, self.bev_h_, self.bev_w_).contiguous()
        else: bev_feats_refined = bev_feats
        step6_time = end_time - start_time
        
        # auxiliry BEV segementation loss
        if self.proposal_layer_latter is not None and self.use_props_supervision:
            bev_mask_logit_latter = self.proposal_layer_latter(bev_feats_refined)
        else: bev_mask_logit_latter = None
        bev_mask_logit = {'former':bev_mask_logit_former,'latter':bev_mask_logit_latter}
        
        # NOTE: here we should make bev_feats back to original cordinates
        bev_feats_refined = bev_feats_refined.permute(0, 1, 3, 2).contiguous()
        step_all_time = step1_time + step2_time + step3_time + step4_time + step5_time + step6_time
        self.recording_fps(step_all_time)
        
        self.draw_gt_pred_rangeview(img_metas, segmentation, bbox_Mask, rangeview_logit.sigmoid()> self.rangeview_foreground.mask_thre_train, rangeview_logit.sigmoid(), eroded=False)
        if bev_mask_logit_former is not None: self.draw_gt_pred_bev(gt_bev_mask, bev_mask_logit_former.sigmoid()>self.proposal_layer_former.mask_thre, bev_mask_logit_former.sigmoid(), img_metas, self.proposal_layer_former.mask_thre, 'former')
        if bev_mask_logit_latter is not None: self.draw_gt_pred_bev(gt_bev_mask, bev_mask_logit_latter.sigmoid()>self.proposal_layer_latter.mask_thre, bev_mask_logit_latter.sigmoid(), img_metas, self.proposal_layer_latter.mask_thre, 'latter')
        self.draw_bev_feature_map(bev_feats_refined, img_metas, bev_feats_name='bev_feats_fusion_refined')
        self.draw_bev_feature_map(bev_feats.permute(0, 1, 3, 2), img_metas, bev_feats_name='bev_feats_fusion')
        self.draw_bev_feature_map(paint_bev, img_metas, bev_feats_name='bev_feats_painting')
        self.draw_bev_feature_map(pts_bev_feats, img_metas, bev_feats_name='bev_feats_pts')
        self.draw_bev_feature_map(img_bev_feats, img_metas, bev_feats_name='bev_feats_img')
        self.draw_bev_feature_map(pts_bev_feats_origin, img_metas, bev_feats_name='bev_feats_pts_origin')
        
        return dict(img_feats=img_feats,
                    pts_feats=[bev_feats_refined],
                    aux_feats_point=[pts_bev_feats],
                    aux_feats_image=[img_bev_feats],
                    pd_depths=depth, 
                    gt_depths=gt_depths,
                    rd_depths=radar_depth,
                    gt_bev_mask=gt_bev_mask,
                    bev_mask_logit=bev_mask_logit,
                    bbox_Mask=bbox_Mask,
                    segmentation=segmentation,
                    rangeview_logit=rangeview_logit,
                    depth_comple=depth_comple,
                    precise_depth=precise_depth)
    def voxelpainting_depth_aware(self, context, pv_logits, depth_logits, points, lidar2img, temperature=1.0):
        B, _, H, W = pv_logits.shape
        device = lidar2img.device
        cam_depth_range = self.grid_config['dbound']
        painted_points = []
        bev_h, bev_w = self.bev_h_, self.bev_w_
        bev_x_min, bev_y_min, bev_z_min, bev_x_max, bev_y_max, bev_z_max = self.point_cloud_range
        context = context*temperature # enlarge
        
        for i in range(B):
            # preparation
            pts = points[i][:, :3]
            pts_hom = torch.cat((pts, torch.ones((pts.shape[0], 1), device=device)), dim=1)  # (N, 4)
            img_pts = torch.matmul(lidar2img[i], pts_hom.t()).t()  # (N, 4)
            img_pts[:, :2] = img_pts[:, :2] / img_pts[:, 2:3]
            depth_values = img_pts[:, 2]

            valid_mask = (img_pts[:, 0] >= 0) & (img_pts[:, 0] < W) & (img_pts[:, 1] >= 0) & (img_pts[:, 1] < H)
            img_pts_int = img_pts[:, :2].long()
            
            # Initialize the feature tensor with zeros, 
            context_features = torch.zeros((pts.shape[0], context.shape[1]+self.pts_dim), device=device, dtype=context.dtype)
            pts_norm = (pts - torch.tensor([bev_x_min, bev_y_min, bev_z_min], device=device)) \
                / torch.tensor([bev_x_max - bev_x_min, bev_y_max - bev_y_min, bev_z_max - bev_z_min], device=device)
            # context_features[:, :3] = pts_norm
            context_features[:, :3] = pts # keep same as radar points
            
            # begin valid point decorated
            valid_img_pts_int = img_pts_int[valid_mask]
            valid_depth_values = depth_values[valid_mask]
            # for grid sample for sub pixel
            valid_img_pts_norm = img_pts[:, :2][valid_mask].clone()
            valid_img_pts_norm[:, 0] = (valid_img_pts_norm[:, 0] / (W - 1)) * 2 - 1
            valid_img_pts_norm[:, 1] = (valid_img_pts_norm[:, 1] / (H - 1)) * 2 - 1
            
            # Extract context features for valid image points
            if self.dataset_type == 'VoD':
                valid_context_features = F.grid_sample(context[i].unsqueeze(0), valid_img_pts_norm.unsqueeze(0).unsqueeze(1), align_corners=True)
                valid_context_features = valid_context_features.squeeze(0).squeeze(1).permute(1, 0)  # (num_points, C)
            if self.dataset_type == 'TJ4D':
                valid_context_features = context[i, :, valid_img_pts_int[:, 1], valid_img_pts_int[:, 0]]
                valid_context_features = valid_context_features.permute(1, 0)  # (num_points, C)
            
            # Get corresponding depth_logits & Weight context_features using the log-transformed depth probabilities
            if self.painting_use_depth_prob:   
                depth_probs = depth_logits[i, :, valid_img_pts_int[:, 1], valid_img_pts_int[:, 0]]
                if self.dataset_type == 'VoD': power_exponent = 1
                if self.dataset_type == 'TJ4D': power_exponent = 2
                power_depth_probs = depth_probs ** power_exponent
                power_depth_probs /= power_depth_probs.sum(dim=0, keepdim=True)  # Normalize
                depth_probs = power_depth_probs
                if self.dataset_type == 'VoD':
                    depth_indices = ((valid_depth_values - cam_depth_range[0]) / cam_depth_range[2])
                    lower_indices = torch.floor(depth_indices).long()
                    upper_indices = torch.ceil(depth_indices).long()
                    lower_indices = torch.clamp(lower_indices, 0, depth_probs.shape[0] - 1)
                    upper_indices = torch.clamp(upper_indices, 0, depth_probs.shape[0] - 1)
                    upper_weight = depth_indices - lower_indices.float()
                    lower_weight = 1 - upper_weight
                    lower_prob_values = depth_probs[lower_indices, range(lower_indices.shape[0])]
                    upper_prob_values = depth_probs[upper_indices, range(upper_indices.shape[0])]
                    depth_prob_values = lower_weight * lower_prob_values + upper_weight * upper_prob_values
                if self.dataset_type == 'TJ4D':
                    depth_indices = ((valid_depth_values - cam_depth_range[0]) / cam_depth_range[2]).long()
                    depth_indices = torch.clamp(depth_indices, 0, depth_probs.shape[0] - 1)
                    depth_prob_values = depth_probs[depth_indices, torch.arange(depth_indices.shape[0], device=device)]
                # re-weight decorated features    
                valid_context_features = valid_context_features * depth_prob_values.unsqueeze(1)
                
            # Assign the computed features to the correct positions
            indices = torch.nonzero(valid_mask).squeeze(1).to(device)
            add_feature = torch.cat((torch.zeros((indices.size(0), self.pts_dim), device=device), valid_context_features), dim=1)
            context_features.index_add_(0, indices, add_feature)
            painted_points.append(context_features)
        
        return painted_points
        
    # train and evaluating process
    
    def simple_test(self, 
                    points, 
                    img_metas, 
                    img=None, 
                    rescale=False, 
                    gt_bboxes_3d=None,
                    gt_labels_3d=None,
                    gt_labels=None,
                    gt_bboxes=None):
        """Test function without augmentaiton."""
        if len(img_metas) !=1: img_metas = [img_metas]
        # preparation for testing
        if gt_bboxes_3d is not None: 
            for i in range(len(img_metas)):
                img_metas[i]['gt_labels'] = gt_labels[i]
                img_metas[i]['gt_bboxes'] = HorizontalBoxes(gt_bboxes[i], in_mode='xyxy')
                img_metas[i]['gt_bboxes_3d'] = gt_bboxes_3d[i].to(gt_labels_3d[i].device)
                img_metas[i]['gt_labels_3d'] = gt_labels_3d[i]
        feature_dict = self.extract_feat(points, img=img, img_metas=img_metas)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats']

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox: # pts means 3D detection
            bbox_pts, outs_pts = self.simple_test_pts(pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox

        if img_feats and self.with_img_bbox: # img means 2D detection
            bbox_img = self.simple_test_img(img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        
        # visualization for test stage 
        threshold = 0.3
        if gt_bboxes_3d is not None and self.use_box3d_supervision:
            if img.dim() == 3 and img.size(0)== 3: img = img.unsqueeze(0)
            if not isinstance(points, list): points = [points]
            self.draw_gt_pred_figures_3d(points, img, gt_bboxes_3d, gt_labels_3d, img_metas, False, threshold, outs_pts=outs_pts)
        else: # vanilla testing method
            self.vis_time_box3d += 1
            if self.vis_time_box3d % self.SAVE_INTERVALS == 0:
                figures_path_det3d = self.figures_path_det3d_test 
                input_img = np.array(img.cpu()).transpose(1,2,0)
                input_img = input_img*self.std[None, None, :] + self.mean[None, None, :]
                pred_bboxes_3d = bbox_pts[0]['boxes_3d']
                pred_scores_3d = bbox_pts[0]['scores_3d']
                pred_bboxes_3d = pred_bboxes_3d[pred_scores_3d>threshold].to('cpu')
                proj_mat = img_metas[0]["final_lidar2img"] # update lidar2img
                img_name = img_metas[0]['filename'].split('/')[-1].split('.')[0]
                # project 3D bboxes to image and get show figures
                if len(pred_bboxes_3d) == 0: pred_bboxes_3d = None
                filename = str(self.vis_time_box3d) + '_' + img_name + '_det3d'
                show_multi_modality_result(img=input_img, gt_bboxes=None, pred_bboxes=pred_bboxes_3d, proj_mat=proj_mat, out_dir=figures_path_det3d, filename=filename, box_mode='lidar', show=False)
            
        return bbox_list

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      img_depth=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        # preparation for loss caculation
        for i in range(len(img_metas)):
            img_metas[i]['gt_labels'] = gt_labels[i]
            img_metas[i]['gt_bboxes'] = HorizontalBoxes(gt_bboxes[i], in_mode='xyxy')
            img_metas[i]['gt_bboxes_3d'] = gt_bboxes_3d[i].to(gt_labels_3d[i].device)
            img_metas[i]['gt_labels_3d'] = gt_labels_3d[i]
        feature_dict = self.extract_feat(points, img=img, img_metas=img_metas)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats']
        gt_depths = feature_dict['gt_depths']
        pd_depths = feature_dict['pd_depths']
        gt_bev_mask = feature_dict['gt_bev_mask']
        bev_mask_logit = feature_dict['bev_mask_logit']
        bbox_Mask = feature_dict['bbox_Mask']
        segmentation = feature_dict['segmentation']
        rangeview_logit = feature_dict['rangeview_logit']
        precise_depth = feature_dict['precise_depth']
        
        # compute for all losses
        losses = dict()
        outs_pts = None
        if self.use_box3d_supervision and gt_bboxes_3d is not None:
            losses_pts, outs_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore)
            losses.update(losses_pts)
        if self.use_depth_supervision and self.depth_net is not None:
            loss_depth = self.depth_net.get_depth_loss(gt_depths, pd_depths, precise_depth, rangeview_logit.sigmoid().detach()) # .detach()
            losses.update(loss_depth)
        if self.use_props_supervision is not None and self.proposal_layer_former is not None:
            if bev_mask_logit['former'] is not None: 
                losses_proposal_former = self.proposal_layer_former.get_bev_mask_loss(gt_bev_mask, bev_mask_logit['former'])
                losses_proposal_former = {f"{key}_former": value for key, value in losses_proposal_former.items()}
                losses.update(losses_proposal_former)
        if self.use_props_supervision is not None and self.proposal_layer_latter is not None:
            if bev_mask_logit['latter'] is not None:
                losses_proposal_latter = self.proposal_layer_latter.get_bev_mask_loss(gt_bev_mask, bev_mask_logit['latter'])
                losses_proposal_latter = {f"{key}_latter": value for key, value in losses_proposal_latter.items()}
                losses.update(losses_proposal_latter)
        if self.use_msk2d_supervision is not None and self.rangeview_foreground is not None:
            losses_proposal = self.rangeview_foreground.get_range_view_mask_loss(bbox_Mask, segmentation, rangeview_logit)
            losses.update(losses_proposal)
        
        # visualization for train stage  
        if gt_bboxes_3d is not None and self.use_box3d_supervision:
            self.draw_gt_pred_figures_3d(points, img, gt_bboxes_3d, gt_labels_3d, img_metas, False, 0.3, outs_pts=outs_pts)

        return losses
    
    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.pts_bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses, outs
    
    # preprocessing for data and others
    
    def preprocessing_information(self, batch_img_metas, device):
        if self.training:
            # all important informations
            batch_size = len(batch_img_metas)
            # gt lidar instances_3d, list of InstancesData
            gt_bboxes_3d = [img_meta['gt_bboxes_3d'] for img_meta in batch_img_metas]
            gt_labels_3d = [img_meta['gt_labels_3d'] for img_meta in batch_img_metas]
            gt_bboxes_2d = [img_meta['gt_bboxes'] for img_meta in batch_img_metas]
            gt_labels_2d = [img_meta['gt_labels'] for img_meta in batch_img_metas]

            # cam_aware: rot, tran, intrin, post_rot, post_tran, _, cam2lidar, focal_length, baseline
            cam_aware = [img_meta['cam_aware'] for img_meta in batch_img_metas]
            merged_tensors = [None] * len(cam_aware[0])
            for i in range(len(cam_aware[0])):
                component = [x[i] for x in cam_aware]
                merged_tensors[i] = torch.stack(component, dim=0)
            cam_aware = merged_tensors
            cam_aware = [x.to(device) for x in cam_aware]
            
            # img_aug_matrix: 4x4 martix of combined post_rot&post_tran of IMG_AUG
            img_aug_matrix = [img_meta['img_aug_matrix'] for img_meta in batch_img_metas]
            img_aug_matrix = torch.tensor(np.stack(img_aug_matrix, axis=0))
            img_aug_matrix = img_aug_matrix.to(device)
            # lidar_aug_matrix same as bda_rot: 4x4 martix of combined post_rot&post_tran of BEV_AUG
            if 'lidar_aug_matrix' in batch_img_metas[0]:
                lidar_aug_matrix = [img_meta['lidar_aug_matrix'] for img_meta in batch_img_metas]
                lidar_aug_matrix = torch.tensor(np.stack(lidar_aug_matrix, axis=0)).to(torch.float32)
                bda_rot = [img_meta['bda_rot'] for img_meta in batch_img_metas]
                bda_rot = torch.tensor(np.stack(bda_rot, axis=0)).to(torch.float32)
            else:
                lidar_aug_matrix = torch.eye(4).unsqueeze(0).repeat(len(batch_img_metas), 1, 1)
                bda_rot = lidar_aug_matrix
            lidar_aug_matrix = lidar_aug_matrix.to(device)
            bda_rot = bda_rot.to(device)
            # create gt_depths from LiDAR data, already processing with IMG_AUG, no need with BEV_AUG
            gt_depths = [img_meta['gt_depths'] for img_meta in batch_img_metas if 'gt_depths' in img_meta]
            gt_depths = torch.stack(gt_depths).unsqueeze(1) # B, 1, H, W
            gt_depths = gt_depths.to(device)
            
            # generate_bev_mask
            gt_bboxes_3d_filtered = [gt_bboxes_3d[i][gt_labels_3d[i] != -1] for i in range(batch_size)] # filter out the ignored labels
            gt_bev_mask = self.generate_bev_mask(gt_bboxes_3d_filtered, batch_size, device, occ_threshold=0.3) # B H W
            gt_bev_mask = gt_bev_mask.to(device)
            
            # re-organize clearly to create NOW lidar2img for project convenience
            batch_img_metas = self.reorganize_lidar2img(batch_img_metas)
            calib = []
            for sample_idx in range(batch_size):
                mat = batch_img_metas[sample_idx]['final_lidar2img']
                mat = torch.Tensor(mat).to(device)
                calib.append(mat)
            final_lidar2img = torch.stack(calib)
            
            # preprocessed seg_mask, pre_inferenced depth_comple
            if 'depth_comple' in batch_img_metas[0].keys():
                depth_comple = [img_meta['depth_comple'] for img_meta in batch_img_metas]
                depth_comple = torch.tensor(np.stack(depth_comple, axis=0)).to(device).unsqueeze(1)
            else: depth_comple = torch.zeros_like(gt_depths).to(device)
            radar_depth = [img_meta['radar_depth'] for img_meta in batch_img_metas]
            radar_depth = torch.tensor(np.stack(radar_depth, axis=0)).to(device).unsqueeze(1)
            radar_depth = radar_depth.to(torch.float32)
            # preprocessed bbox_Mask and segmentation for msk2D supervison, NOTE: downsampled
            h, w = batch_img_metas[0]['img_shape']
            h_down, w_down = h // self.downsample, w // self.downsample
            if 'segmentation' in batch_img_metas[0].keys():
                segmentation = [img_meta['segmentation'].astype(np.float32) for img_meta in batch_img_metas]
                segmentation = torch.tensor(np.stack(segmentation, axis=0), dtype=torch.float32).to(device).unsqueeze(1)
                segmentation = F.interpolate(segmentation, (h_down, w_down), mode='bilinear', align_corners=True)
            else: segmentation = torch.zeros((len(batch_img_metas), 1, h_down, w_down), dtype=torch.float32).to(device)
            bbox_Mask = [img_meta['bbox_Mask'] for img_meta in batch_img_metas]
            bbox_Mask = torch.tensor(np.stack(bbox_Mask, axis=0)).to(device).unsqueeze(1)
            bbox_Mask = F.interpolate(bbox_Mask, (h_down, w_down), mode='bilinear', align_corners=True)
        else:
            # gt_bboxes_3d, gt_labels_3d, gt_bboxes_2d, gt_labels_2d, gt_depths = [], [], [], [], []
            batch_img_metas = batch_img_metas[0]
            h, w = batch_img_metas['img_shape']
            if 'gt_bboxes_3d' in batch_img_metas:
                gt_bboxes_3d = [batch_img_metas['gt_bboxes_3d']]
            else: gt_bboxes_3d = []
            if 'gt_labels_3d' in batch_img_metas:
                gt_labels_3d = [batch_img_metas['gt_labels_3d']]
            else: gt_labels_3d = []
            if 'gt_bboxes_2d' in batch_img_metas:
                gt_bboxes_2d = [batch_img_metas['gt_bboxes_2d']]
            else: gt_bboxes_2d = []
            if 'gt_labels_2d' in batch_img_metas:
                gt_labels_2d = [batch_img_metas['gt_labels_2d']]
            else: gt_labels_2d = []
            if 'gt_depths' in batch_img_metas:
                gt_depths = [batch_img_metas['gt_depths']]
                gt_depths = torch.stack(gt_depths).unsqueeze(1) # B, 1, H, W
                gt_depths = gt_depths.to(device)
            else: gt_depths = torch.zeros((1, 1, h, w)).to(device)
            H, W = batch_img_metas['img_shape']
            cam_aware = batch_img_metas['cam_aware']
            cam_aware = [[x.to(device)] for x in cam_aware]
            cam_aware = [torch.stack(x, dim=0) for x in cam_aware]
            img_aug_matrix = [batch_img_metas['img_aug_matrix']]
            img_aug_matrix = torch.tensor(np.stack(img_aug_matrix, axis=0))
            img_aug_matrix = img_aug_matrix.to(device)
            if 'lidar_aug_matrix' in batch_img_metas:
                lidar_aug_matrix = [batch_img_metas['lidar_aug_matrix']]
                lidar_aug_matrix = torch.tensor(np.stack(lidar_aug_matrix, axis=0)).to(torch.float32)
                bda_rot = [batch_img_metas['bda_rot'] ]
                bda_rot = torch.tensor(np.stack(bda_rot, axis=0)).to(torch.float32)
            else:
                lidar_aug_matrix = torch.eye(4).unsqueeze(0)
                bda_rot = lidar_aug_matrix
            lidar_aug_matrix = lidar_aug_matrix.to(device)
            bda_rot = bda_rot.to(device)
            gt_bev_mask = torch.zeros((1,1,self.bev_h_,self.bev_w_)).to(device)
            batch_img_metas = self.reorganize_lidar2img([batch_img_metas]) # begin list again
            calib = []
            mat = batch_img_metas[0]['final_lidar2img']
            mat = torch.Tensor(mat).to(device)
            final_lidar2img = torch.stack([mat])
            if 'depth_comple' in batch_img_metas[0].keys():
                depth_comple = [batch_img_metas[0]['depth_comple']] if isinstance(batch_img_metas, list) else [batch_img_metas['depth_comple']]
                depth_comple = torch.tensor(np.stack(depth_comple, axis=0)).to(device).unsqueeze(1)
            else: depth_comple = torch.zeros((1, 1, H, W)).to(device)
            radar_depth = [batch_img_metas[0]['radar_depth']] if isinstance(batch_img_metas, list) else [batch_img_metas['depth_comple']]
            radar_depth = torch.tensor(np.stack(radar_depth, axis=0)).to(device).unsqueeze(1)
            radar_depth = radar_depth.to(torch.float32)
            h, w = batch_img_metas[0]['img_shape']
            h_down, w_down = h // self.downsample, w // self.downsample
            if 'segmentation' in batch_img_metas[0].keys():
                segmentation = [batch_img_metas[0]['segmentation'].astype(np.float32)] if isinstance(batch_img_metas, list) else [batch_img_metas['segmentation']]
                segmentation = torch.tensor(np.stack(segmentation, axis=0)).to(device).unsqueeze(1)
                segmentation = F.interpolate(segmentation, (h_down, w_down), mode='bilinear', align_corners=True)
            else: segmentation = torch.zeros((1, 1, h_down, w_down)).to(device)
            bbox_Mask = [batch_img_metas[0]['bbox_Mask']] if isinstance(batch_img_metas, list) else [batch_img_metas['bbox_Mask']]
            bbox_Mask = torch.tensor(np.stack(bbox_Mask, axis=0)).to(device).unsqueeze(1)
            bbox_Mask = F.interpolate(bbox_Mask, (h_down, w_down), mode='bilinear', align_corners=True)
        return batch_img_metas, gt_bboxes_3d, gt_labels_3d, gt_bboxes_2d, gt_labels_2d, depth_comple, bbox_Mask, segmentation, radar_depth, \
            cam_aware, img_aug_matrix, lidar_aug_matrix, bda_rot, gt_depths, gt_bev_mask, final_lidar2img

    def reorganize_lidar2img(self, batch_input_metas):
        """add 'lidar2img' transformation matrix into batch_input_metas.

        Args:
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.
        Returns:
            batch_input_metas (list[dict]): Meta info with lidar2img added
        """
        for img_metas in batch_input_metas:
            final_cam2img = copy.deepcopy(img_metas['cam2img'])
            final_lidar2img = copy.deepcopy(img_metas['lidar2img'])
            
            # same as visualization in BEVAug3D 
            rots, trans, intrins, post_rots, post_trans = img_metas['cam_aware'][:5]
            final_cam2img[:2, :3] = post_rots[:2, :2] @ final_cam2img[:2, :3]
            final_cam2img[:2, 2] = post_trans[:2] + final_cam2img[:2, 2]
            final_lidar2img = final_cam2img @ img_metas['lidar2cam']
            final_lidar2img = final_lidar2img @ np.linalg.inv(img_metas['lidar_aug_matrix'])
            img_metas['final_lidar2img'] = final_lidar2img

        return batch_input_metas
    
    def generate_bev_mask(self, gt_bboxes_3d, batch_size, device, occ_threshold):
        # As long as it is occupied, it is 1
        gt_bev_mask = []
        if len(gt_bboxes_3d) != 0:
            bev_cell_size = torch.tensor(self.bev_cell_size).to(device)
            for bsid in range(len(gt_bboxes_3d)):
                bev_mask = torch.zeros(self.bev_grid_shape)
                bbox_corners = gt_bboxes_3d[bsid].corners[:, [0,2,4,6],:2] # bev corners
                num_rectangles = bbox_corners.shape[0]
                bbox_corners[:,:,0] = (bbox_corners[:,:,0] - self.xbound[0])/bev_cell_size[0] # id_num, 4, 2
                bbox_corners[:,:,1] = (bbox_corners[:,:,1] - self.ybound[0])/bev_cell_size[1] # id_num, 4, 2
                
                # precise bur slow method
                grid_min = torch.clip(torch.floor(torch.min(bbox_corners, axis=1).values).to(torch.int64), 0, self.bev_grid_shape[0] - 1)
                grid_max = torch.clip(torch.ceil (torch.max(bbox_corners, axis=1).values).to(torch.int64), 0, self.bev_grid_shape[1] - 1)
                possible_mask_h_all = torch.cat([grid_min[:, 0:1], grid_max[:, 0:1]], dim=1).tolist()
                possible_mask_w_all = torch.cat([grid_min[:, 1:2], grid_max[:, 1:2]], dim=1).tolist()
                for n in range(num_rectangles):
                    clock_corners = bbox_corners[n].cpu().numpy()[(0,1,3,2), :]
                    poly = Polygon(clock_corners)
                    h_list = possible_mask_h_all[n]; h_list = np.arange(h_list[0] - 1, h_list[1] + 1, 1); h_list = np.clip(h_list, 0, self.bev_grid_shape[0] - 1)
                    w_list = possible_mask_w_all[n]; w_list = np.arange(w_list[0] - 1, w_list[1] + 1, 1); w_list = np.clip(w_list, 0, self.bev_grid_shape[1] - 1)
                    for i in h_list:
                        for j in w_list:
                            cell_center = np.array([i + 0.5, j + 0.5])
                            cell_poly = box(i, j, i + 1, j + 1)
                            if poly.contains(Point(cell_center)):
                                bev_mask[i, j] = True
                            else:
                                intersection = cell_poly.intersection(poly)
                                if (intersection.area / cell_poly.area) > occ_threshold: bev_mask[i, j] = True
                # coarse but quick method
                # for i in range(num_rectangles):
                #     bev_mask[grid_min[i, 0]:grid_max[i, 0], grid_min[i, 1]:grid_max[i, 1]] = True
                # save_image(bev_mask[None,None,:,:]*0.99, 'gt_bev_mask.png')
                gt_bev_mask.append(bev_mask)
            gt_bev_mask = torch.stack(gt_bev_mask, dim=0).unsqueeze(1) # B 1 H W
        else:
            gt_bev_mask = torch.zeros((batch_size, 1, self.bev_grid_shape[0], self.bev_grid_shape[1]))
        gt_bev_mask = gt_bev_mask.to(torch.bool)
        return gt_bev_mask
    
    def recording_fps(self, step_all_time):
        self.record_fps['num'] += 1
        self.record_fps['time'] += step_all_time
        if not self.training and self.record_fps['num'] % 50 == 0: 
            print(' FPS: %.2f'%(1.0/step_all_time))
        if not self.training and self.record_fps['num'] == 1296 and not self.training: 
            print(' FINAL VOD FPS: %.2f'%(1296/self.record_fps['time']))
            
    @master_only
    def draw_gt_pred_figures_3d(self, points, imgs, gt_bboxes_3ds, gt_labels_3ds, img_metas, rescale=False, threshold=0.3, **kwargs):
        # if training we should decode the bbox from features 'outs_pts' first
        self.vis_time_box3d += 1
        if not self.vis_time_box3d % self.SAVE_INTERVALS == 0: return
        # filter out the ignored labels
        if self.training: figures_path_det3d = self.figures_path_det3d_train
        else: figures_path_det3d = self.figures_path_det3d_test
        gt_bboxes_3ds = [gt_bboxes_3ds[i][gt_labels_3ds[i]!= -1] for i in range(len(img_metas))]
        outs_pts = kwargs['outs_pts']
        if outs_pts is not None:
            bbox_list = self.pts_bbox_head.get_bboxes(*outs_pts, img_metas, rescale=False)
            bbox_list = [bbox3d2result(bboxes, scores, labels)for bboxes, scores, labels in bbox_list]
        else: bbox_list = None
                
        # starting visualization
        for i in range(imgs.shape[0]): # batch size
            # preparation
            input_img = np.array(imgs[i].cpu()).transpose(1,2,0)
            input_img = input_img*self.std[None, None, :] + self.mean[None, None, :]
            pred_bboxes_3d = bbox_list[i]['boxes_3d'] if bbox_list is not None else None
            pred_scores_3d = bbox_list[i]['scores_3d'] if bbox_list is not None else None
            pred_bboxes_3d = pred_bboxes_3d[pred_scores_3d>threshold].to('cpu') if bbox_list is not None else None
            gt_bboxes_3d = gt_bboxes_3ds[i].to('cpu')
            proj_mat = img_metas[i]["final_lidar2img"] # update lidar2img
            img_name = img_metas[i]['filename'].split('/')[-1].split('.')[0]
            # project 3D bboxes to image and get show figures
            if pred_bboxes_3d is not None:
                if len(pred_bboxes_3d) == 0: pred_bboxes_3d = None
                
            # draw in image view
            filename = str(self.vis_time_box3d) + '_' + img_name + '_det3d'
            result_path = figures_path_det3d; mmcv.mkdir_or_exist(result_path)
            # show_multi_modality_result(img=input_img, gt_bboxes=gt_bboxes_3d, pred_bboxes=pred_bboxes_3d, proj_mat=proj_mat, out_dir=figures_path_det3d, filename=filename, box_mode='lidar', show=False)
            # draw in bev view
            save_path = os.path.join(figures_path_det3d, str(self.vis_time_box3d) + '_' + img_name + '_det3d_bev.png')
            save_path_paper = os.path.join(figures_path_det3d, str(self.vis_time_box3d) + '_' + img_name + '_det3d_bev_paper.png')
            point = points[i].cpu().detach().numpy()[:, :3]
            pd_bbox_corners = pred_bboxes_3d.corners[:, [0,2,4,6],:2].numpy()[:, (0,1,3,2), :] if pred_bboxes_3d is not None else None
            gt_bbox_corners = gt_bboxes_3d.corners[:, [0,2,4,6],:2].numpy()[:, (0,1,3,2), :] if gt_bboxes_3d is not None else None
            draw_bev_pts_bboxes(point, gt_bbox_corners, pd_bbox_corners, save_path=save_path, xlim=self.xlim, ylim=self.ylim) 
            # for paper figures
            tmp_img_true = custom_draw_lidar_bbox3d_on_img(gt_bboxes_3d, input_img, proj_mat, img_metas, color=(61, 102, 255), thickness=2, scale_factor=3)
            tmp_img_pred = custom_draw_lidar_bbox3d_on_img(pred_bboxes_3d, input_img, proj_mat, img_metas, color=(241, 101, 72), thickness=2, scale_factor=3)
            tmp_img_alls = custom_draw_lidar_bbox3d_on_img(pred_bboxes_3d, tmp_img_true, proj_mat, img_metas, color=(241, 101, 72), thickness=2, scale_factor=3)
            mmcv.imwrite(tmp_img_true, os.path.join(result_path, f'{filename}_gt.png'))
            mmcv.imwrite(tmp_img_pred, os.path.join(result_path, f'{filename}_pred.png'))
            mmcv.imwrite(tmp_img_alls, os.path.join(result_path, f'{filename}.png'))
            draw_paper_bboxes(point, gt_bbox_corners, pd_bbox_corners, save_path=save_path_paper, xlim=self.xlim, ylim=self.ylim)

    @master_only
    def draw_gt_pred_bev(self, gt_bev_mask, bev_mask, bev_mask_logit_sigmoid, img_metas, mask_thre, suffix='former'):
        if suffix == 'former': self.vis_time_bev2d += 1
        if not self.vis_time_bev2d % self.SAVE_INTERVALS == 0: return
        if self.training: figures_path_bev2d = self.figures_path_bev2d_train
        else: figures_path_bev2d = self.figures_path_bev2d_test
        
        bev1 = torch.rot90(gt_bev_mask, k=1, dims=(2, 3))
        bev2 = torch.rot90(bev_mask, k=1, dims=(2, 3))
        bev3 = torch.rot90(bev_mask_logit_sigmoid, k=1, dims=(2, 3))
        bev4 = torch.rot90(bev_mask_logit_sigmoid > mask_thre, k=1, dims=(2, 3))
        b, _, h, w = bev1.shape
        frame_1 = 0.5*torch.ones((1, h, 5)).to(bev_mask_logit_sigmoid.device)
        for i in range(bev_mask.shape[0]):
            img_name = img_metas[i]['filename'].split('/')[-1].split('.')[0]
            save_bev = torch.cat([frame_1, bev1[i], frame_1, bev2[i], frame_1, bev3[i], frame_1, bev4[i], frame_1], dim=2)*0.99
            frame_2 = 0.5*torch.ones((1, 5, save_bev.shape[2])).to(bev_mask_logit_sigmoid.device)
            save_bev = torch.cat([frame_2, save_bev, frame_2], dim=1)
            save_image(save_bev, os.path.join(figures_path_bev2d, str(self.vis_time_bev2d) + '_' + img_name + '_bev2d_'+ suffix +'.png'))
            
    @master_only
    def draw_bev_feature_map(self, bev_feats, img_metas, bev_feats_name='bev_feats_fusion'):
        if bev_feats_name=='bev_feats_fusion_refined': self.vis_time_bevnd += 1
        if not self.vis_time_bevnd % self.SAVE_INTERVALS == 0: return
        if self.training: figures_path_bevnd = self.figures_path_bevnd_train
        else: figures_path_bevnd = self.figures_path_bevnd_test
            
        b, _, h, w = bev_feats.shape 
        # bev_feats = bev_feats.mean(1).unsqueeze(1) # using mean
        bev_feats_show = bev_feats.max(1, keepdim=True).values # using max
        # bev_feats_show = torch.rot90(bev_feats_show, k=2, dims=(2, 3))\
        bev_feats_show = torch.flip(bev_feats_show, [2]) # horizontal flip for consistency to gt bev bbox
        for i in range(bev_feats.shape[0]):
            img_name = img_metas[i]['filename'].split('/')[-1].split('.')[0]
            bev_feats_tmp = bev_feats_show[i:i+1, :, :, :]
            bev_feats_tmp = (bev_feats_tmp - bev_feats_tmp.min())/(bev_feats_tmp.max() - bev_feats_tmp.min())
            # bev_feats_tmp = (bev_feats_tmp - 0.75)/(1.00 - 0.75)
            if bev_feats_name == 'bev_feats_img': bev_feats_tmp = bev_feats_tmp*25
            bev_feats_tmp_np = bev_feats_tmp.squeeze().cpu().detach().numpy()
            bev_feats_tmp_colored = plt.cm.viridis(bev_feats_tmp_np)[..., :3] 
            bev_feats_tmp_colored = torch.tensor(bev_feats_tmp_colored).permute(2, 0, 1).unsqueeze(0)
            save_image(bev_feats_tmp_colored, os.path.join(figures_path_bevnd, str(self.vis_time_bevnd) + '_' + img_name + '_' + bev_feats_name + '.png'))
    
    @master_only
    def draw_gt_pred_rangeview(self, img_metas, segs, gts, preds, sigmoids, eroded=False):
        if not eroded: self.vis_time_range += 1
        if not self.vis_time_range % self.SAVE_INTERVALS == 0: return
        if self.training: figures_path_range = self.figures_path_range_train
        else: figures_path_range = self.figures_path_range_test
        
        b, _, h, w = gts.shape
        frame_1 = 0.5*torch.ones((1, 5, w)).to(preds.device)
        for i in range(gts.shape[0]):
            img_name = img_metas[i]['filename'].split('/')[-1].split('.')[0]
            seg = segs[i]; gt = gts[i]; pred = preds[i]; sigmoid = sigmoids[i]
            save_range = torch.cat([frame_1, seg, frame_1, gt, frame_1, pred, frame_1, sigmoid, frame_1], dim=1)
            frame_2 = 0.5*torch.ones((1, save_range.shape[1], 5)).to(pred.device)
            save_range = torch.cat([frame_2, save_range, frame_2], dim=2)
            if not eroded: save_image(save_range, os.path.join(figures_path_range, str(self.vis_time_range) + '_' + img_name + '_range.png'))
            else: save_image(save_range, os.path.join(figures_path_range, str(self.vis_time_range) + '_' + img_name + '_range_eroded.png'))
          
    @master_only     
    def draw_pts_completion(self, img_metas, gt_points, pd_points, gt_bboxes_3d=None, gt_labels_3d=None, plot_mode='distance', points_type='pointpainting'):
        if points_type == 'voxelpainting': self.vis_time_point += 1
        if not self.vis_time_point % self.SAVE_INTERVALS == 0: return
        if self.training: figures_path_point = self.figures_path_point_train
        else: figures_path_point = self.figures_path_point_test
        
        for i in range(len(img_metas)):
            img_name = img_metas[i]['filename'].split('/')[-1].split('.')[0]
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            
            for ax, points, title in zip(axes, [gt_points, pd_points], ['Raw Points', 'Virtual Points']):
                points = points[i][(self.xlim[0]<=points[i][:,0]) & (points[i][:,0]<=self.xlim[1]) & \
                    (self.ylim[0]<=points[i][:,1]) & (points[i][:,1]<=self.ylim[1])]
                ax.set_xlim(self.xlim[0], self.xlim[1])
                ax.set_ylim(self.ylim[0], self.ylim[1])
                ax.autoscale(False)

                # plot points
                points = points.cpu().detach().numpy()
                x = points[:, 0]
                y = points[:, 1]
                if plot_mode == 'distance': 
                    intensities = np.clip(np.sqrt(x**2 + y**2) / 60, 0, 1)
                    colors = plt.cm.gray(intensities)
                if plot_mode == 'RCS': 
                    norm_max = np.max(gt_points[i].cpu().detach().numpy()[:, 3])
                    norm_min = np.min(gt_points[i].cpu().detach().numpy()[:, 3])
                    intensities = np.clip((points[:, 3]-norm_min)/(norm_max-norm_min), 0, 1)
                    colors = plt.cm.jet(intensities)
                if plot_mode == 'v_r_compensated': 
                    norm_max = np.max(gt_points[i].cpu().detach().numpy()[:, 4])
                    norm_min = np.min(gt_points[i].cpu().detach().numpy()[:, 4])
                    intensities = np.clip((points[:, 4]-norm_min)/(norm_max-norm_min), 0, 1)
                    colors = plt.cm.jet(intensities)
                if plot_mode == 'logits': 
                    norm_max = np.max(points[:, -1])
                    norm_min = np.min(points[:, -1])
                    intensities = np.clip((points[:, -1]-norm_min)/(norm_max-norm_min), 0, 1)
                    intensities = 1 - intensities
                    intensities = intensities*0.6 + 0.2 # 0.2 - 0.8
                    colors = plt.cm.gray(intensities)
                if plot_mode == 'context_pointpainting':
                    context = np.sum(points[:, 5:], axis=-1)
                    norm_max = np.max(np.sum(gt_points[i].cpu().detach().numpy()[:, 5:], axis=-1))
                    norm_min = np.min(np.sum(gt_points[i].cpu().detach().numpy()[:, 5:], axis=-1))
                    intensities = np.clip((context-norm_min)/(norm_max-norm_min), 0, 1)
                    colors = plt.cm.jet(intensities)
                if plot_mode == 'context_voxelpainting': 
                    context = np.sum(points[:, 5:], axis=-1)
                    norm_max = np.max(np.sum(gt_points[i].cpu().detach().numpy()[:, 5:], axis=-1))
                    norm_min = np.min(np.sum(gt_points[i].cpu().detach().numpy()[:, 5:], axis=-1))
                    intensities = np.clip((context-norm_min)/(norm_max-norm_min), 0, 1)
                    intensities = 1 - intensities
                    intensities = intensities*0.6 + 0.2 # 0.2 - 0.8
                    colors = plt.cm.gray(intensities)
                    sorted_indices = np.argsort(-intensities)
                    x = x[sorted_indices]
                    y = y[sorted_indices]
                    colors = colors[sorted_indices]
                ax.scatter(x, y, c=colors, s=15) # alpha=0.5

                # plot bboxes
                if gt_bboxes_3d is not None:
                    if len(gt_bboxes_3d) != 0:
                        gt_bboxes_3d_filtered = gt_bboxes_3d[i][gt_labels_3d[i] != -1]
                        gt_bbox_corners = gt_bboxes_3d_filtered.corners[:, [0,2,4,6],:2]
                        gt_bbox_corners = gt_bbox_corners.cpu().detach().numpy()[:, (0,1,3,2), :] # clock_corners
                        for bbox in gt_bbox_corners:
                            polygon = patches.Polygon(bbox, closed=True, edgecolor='red', linewidth=1, fill=False)
                            ax.add_patch(polygon)
                        
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_title(f'Point cloud and bboxes under BEV - {title}')
                ax.grid(True)
            
            save_path = os.path.join(figures_path_point, str(self.vis_time_point) + '_' + img_name + '_' + points_type +'.png')
            plt.savefig(save_path)
            plt.close(fig)