import matplotlib.pyplot as plt
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib
from matplotlib.path import Path
import cv2
import copy

def custom_draw_lidar_bbox3d_on_img(bboxes3d,
                             raw_img,
                             lidar2img_rt,
                             img_metas,
                             color=(0, 255, 0),
                             thickness=1,
                             scale_factor=2):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`LiDARInstance3DBoxes`):
            3d bbox in lidar coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        img_metas (dict): Useless here.
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    img = raw_img.copy()
    if bboxes3d is None: 
        return img
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3),
         np.ones((num_bbox * 8, 1))], axis=-1)
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = pts_4d @ lidar2img_rt.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return custom_plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness, scale_factor)

def custom_plot_rect3d_on_img(img,
                       num_rects,
                       rect_corners,
                       color=(0, 255, 0),
                       thickness=1,
                       scale_factor=2):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    # Scale the image up for super-sampling
    h, w = img.shape[:2]
    img_large = cv2.resize(img, (w * scale_factor, h * scale_factor))
    thickness_large = thickness * scale_factor
    
    # Scale corners up by the same factor
    rect_corners_large = rect_corners * scale_factor
    
    # Draw the rectangles
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    for i in range(num_rects):
        corners = rect_corners_large[i].astype(np.int32)
        for start, end in line_indices:
            cv2.line(img_large, (corners[start, 0], corners[start, 1]),
                     (corners[end, 0], corners[end, 1]), color, thickness_large,
                     cv2.LINE_AA)

    # Resize the image back to the original size
    img = cv2.resize(img_large, (w, h), interpolation=cv2.INTER_AREA)
    # cv2.imwrite('1.png', img.astype(np.uint8))
    
    return img.astype(np.uint8)

def draw_bev_pts_bboxes(points, gt_bbox_corners=None, pd_bbox_corners=None, save_path=None, xlim=[0, 51.2], ylim=[-25.6, 25.6]):
        
    figsize_x = xlim[1] - xlim[0]
    figsize_y = ylim[1] - ylim[0]
    fig, ax = plt.subplots(figsize=(figsize_x/6, figsize_y/6))
    points = points[(xlim[0]<=points[:,0]).astype(bool) & (points[:,0]<=xlim[1]).astype(bool)]
    points = points[(ylim[0]<=points[:,1]).astype(bool) & (points[:,1]<=ylim[1]).astype(bool)]
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.autoscale(False)
    
    # plot points
    x = points[:, 0]
    y = points[:, 1]
    distances = np.sqrt(x**2 + y**2)
    normalized_distances = np.clip(distances / 60, 0, 1)
    colors = plt.cm.gray(normalized_distances)
    ax.scatter(x, y, c=colors, s=3)
    
    # plot bboxes
    if gt_bbox_corners is not None:
        for bbox in gt_bbox_corners:
            polygon = patches.Polygon(bbox, closed=True, edgecolor='red', linewidth=1, fill=False)
            ax.add_patch(polygon)
    if pd_bbox_corners is not None:
        for bbox in pd_bbox_corners:
            polygon = patches.Polygon(bbox, closed=True, edgecolor='blue', linewidth=1, fill=False)
            ax.add_patch(polygon)
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('point cloud and bboxes under BEV')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close(fig)

def draw_paper_bboxes(points, gt_bbox_corners=None, pd_bbox_corners=None, save_path=None, xlim=[0, 51.2], ylim=[-25.6, 25.6]):
    
    figsize_x = xlim[1] - xlim[0]
    figsize_y = ylim[1] - ylim[0]
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))
    # fig.patch.set_facecolor('black')
    # ax.set_facecolor('black')
    
    points = points[(xlim[0]<=points[:,0]).astype(bool) & (points[:,0]<=xlim[1]).astype(bool)]
    points = points[(ylim[0]<=points[:,1]).astype(bool) & (points[:,1]<=ylim[1]).astype(bool)]
    
    # Rotate the coordinates by 90 degrees clockwise
    x = -points[:, 1]  # Swap x and y, negate new x to rotate clockwise
    y = points[:, 0]   # Swap x and y

    ax.set_xlim(-ylim[1], -ylim[0])   # X-axis corresponds to the original Y-axis, reversed
    ax.set_ylim(xlim[0], xlim[1])     # Y-axis corresponds to the original X-axis
    ax.autoscale(False)
    
    # Generate colors for all points based on distance
    distances = np.sqrt(x**2 + y**2)
    normalized_distances = np.clip(distances / 60, 0, 1)
    colors = plt.cm.gray(normalized_distances)  # Generate grayscale colors

    # Plot and fill pd bboxes with yellow
    if gt_bbox_corners is not None:
        for bbox in gt_bbox_corners:
            rotated_bbox = np.array([[-corner[1], corner[0]] for corner in bbox])
            polygon = patches.Polygon(rotated_bbox, closed=True, edgecolor='orange', linewidth=10, facecolor=(1, 1, 0), fill=True)
            ax.add_patch(polygon)
            centroid = np.mean(rotated_bbox, axis=0)
            direction = np.mean([rotated_bbox[0], rotated_bbox[3]], axis=0)
            ax.plot([centroid[0], direction[0]], [centroid[1], direction[1]], color='orange', linewidth=10)

    # Plot gt bboxes
    if pd_bbox_corners is not None:
        for bbox in pd_bbox_corners:
            rotated_bbox = np.array([[-corner[1], corner[0]] for corner in bbox])
            polygon = patches.Polygon(rotated_bbox, closed=True, edgecolor='darkgreen', linewidth=10, fill=False)
            ax.add_patch(polygon)
            centroid = np.mean(rotated_bbox, axis=0)
            direction = np.mean([rotated_bbox[1], rotated_bbox[2]], axis=0)
            ax.plot([centroid[0], direction[0]], [centroid[1], direction[1]], color='darkgreen', linewidth=10)
    
    # Check if points are inside gt_bbox_corners and plot them
    if gt_bbox_corners is not None:
        inside_gt = np.zeros(points.shape[0], dtype=bool)
        for bbox in gt_bbox_corners:
            rotated_bbox = np.array([[-corner[1], corner[0]] for corner in bbox])
            path = Path(rotated_bbox)
            inside_gt |= path.contains_points(np.vstack((x, y)).T)
        
        ax.scatter(x[inside_gt], y[inside_gt], c=np.array([[1, 0, 0]]), s=200)  # Red points
        # ax.scatter(x[~inside_gt], y[~inside_gt], c=colors[~inside_gt], s=1)  # Normal points
        ax.scatter(x[~inside_gt], y[~inside_gt], c=np.array([[0.3, 0.3, 0.3]]), s=200)  # Normal points
        # ax.scatter(x, y, c=np.array([[0.5, 0.5, 0.5]]), s=1) 
    else:
        ax.scatter(x, y, c=colors, s=3)
    
    # Draw a red triangle at the origin
    triangle = patches.Polygon([[0, 1.5], [-1.5, -1.5], [1.5, -1.5]], closed=True, edgecolor='red', facecolor='red')
    ax.add_patch(triangle)
    
    # ax.invert_xaxis() # Flip the x-axis
    ax.axis('off')  # Remove the axes
    plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
def box3d2x0y0wh(boxes_3d):
	# BEVDet/CenterPoints
    import numpy as np
    n = boxes_3d.shape[0]
    box2d = np.zeros((n,4))
    # 3dbox --> xywh
    box2d[:,:2] = boxes_3d[:,:2]
    box2d[:,2] = boxes_3d[:,3]
    box2d[:,3] = boxes_3d[:,4] # 2xywh
    # 
    # xyxy = np.ones_like(box2d)
    box2d[:,0] = box2d[:, 0] - box2d[:, 2] / 2 
    box2d[:,1] = box2d[:, 1] + box2d[:, 3] / 2 # NOTE: 左下角点
    
    return box2d

def box3d2x0y0wh_2(boxes_3d):
    # FCOS3D: front view--> BEV
    import numpy as np
    n = boxes_3d.shape[0]
    box2d = np.zeros((n,4))
    # 3dbox --> xywh 左下角点
    box2d[:, 0] = boxes_3d[:, 0]
    box2d[:, 1] = boxes_3d[:, 2]
    box2d[:, 2] = boxes_3d[:, 4]
    box2d[:, 3] = boxes_3d[:, 5] # 2xywh
    # 
    # xyxy = np.ones_like(box2d)
    box2d[:,0] = box2d[:, 0] - box2d[:, 2] / 2 
    box2d[:,1] = box2d[:, 1] + box2d[:, 3] / 2 # NOTE
    
    return box2d

# 根据坐标作图
def draw_boxes(pred_boxes_3d, target_boxes_3d, path):
    # pred_boxes xywh
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    pred_boxes = box3d2x0y0wh(pred_boxes_3d)
    target_boxes = box3d2x0y0wh(target_boxes_3d)

    fig, ax = plt.subplots()
    ax.plot()
    # ax.add_patch(patches.Rectangle((1, 1),0.5,0.5,edgecolor = 'blue',facecolor = 'red',fill=True) )
    
    #
    for index, coord in enumerate(pred_boxes):
        rect = patches.Rectangle((coord[0], coord[1]), coord[2], coord[3], 
        						linewidth=1, edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    for index, coord in enumerate(target_boxes):
        rect = patches.Rectangle((coord[0], coord[1]), coord[2], coord[3], 
        						linewidth=1, edgecolor='g',facecolor='none')
        ax.add_patch(rect)
    
    # plt.legend(loc='best',edgecolor='g')
    if os.path.exists(path):
        os.remove(path)
    fig.savefig(path, dpi=90, bbox_inches='tight')
    # print(0)
    plt.close(fig)
    print('Successfully saved')
    

def draw_pts(points, save_path, point_size=20, show=False):
    '''
    points: [N,3+c]
    '''
    assert len(points.shape) == 2
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    points = points.copy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # point_range = range(0, points.shape[0], skip) # skip points to prevent crash
    point_range = range(0, points.shape[0])
    ax.scatter(points[point_range, 0],   # x
            points[point_range, 1],   # y
            points[point_range, 2],   # z
            c=points[point_range, 2], # height data for color
            cmap=plt.get_cmap("Spectral"),
            marker="o", s=point_size)
    ax.axis('auto')  # {equal, scaled}
    if show:
        plt.show()

    if save_path is not None:
        fig.savefig(save_path, dpi=90, bbox_inches='tight')
    plt.close(fig)

def colorize(value, vmin=None, vmax=None, cmap='jet', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img


    # ######################################### recording ###################################################################
    # if self.use_depth2lidar and depth is not None and rangeview_mask is not None:
    # ###################################### check if vanilla gt unprojection is right ######################################
    # gravity_center = [gt_bboxes_3d[i][gt_labels_3d[i]!=-1].gravity_center for i in range(B)]
    # assert depth_comple.shape[2] == H and depth_comple.shape[3] == W
    # depths_vis, depth_output = self.project_points(depth_comple, gravity_center, final_lidar2img)
    # save_image(torch.cat([0.99*segmentation.repeat(1,3,1,1), gt_depths.repeat(1,3,1,1)/20.0, depths_vis, img], dim=2), 'check.png')
    # unprojection = torch.inverse(final_lidar2img.cpu()).to(depth.device)
    # complete_points = self.unproject_points(gt_depths, unprojection, segmentation.bool().detach(), crop_top_rate=0.0)
    # self.draw_pts_completion(img_metas, points, complete_points, gt_bboxes_3d)
    # ################################## check if domnsample gt unprojection is right ########################################
    # down = self.downsample
    # matrix = torch.eye(4).to(depth.device)
    # matrix[0, 0] = matrix[0, 0] / down
    # matrix[1, 1] = matrix[1, 1] / down
    # matrix = matrix.repeat(B, 1, 1)
    # projection = matrix @ final_lidar2img
    # unprojection = torch.inverse(projection.cpu()).to(depth.device)
    # segmentat_down = F.interpolate(segmentation, (H//down, W//down), mode='bilinear', align_corners=True) 
    # gt_depths_down = get_downsample_depths_torch(depth_comple, down=down, processing='min')
    # complete_points = self.unproject_points(gt_depths_down, unprojection, segmentat_down.bool().detach(), crop_top_rate=0.0)
    # self.draw_pts_completion(img_metas, points, complete_points, gt_bboxes_3d)
    # ############################################## begin our unprojection ##################################################
    # begin downsampled image unprojection with our predicted rangeview_mask
    # cam_depth_range = self.grid_config['dbound']
    # raw_depth = torch.arange(cam_depth_range[0], cam_depth_range[1], cam_depth_range[2]).to(depth.device)
    # precise_depth = torch.sum(raw_depth.view(1,-1,1,1)*depth, dim=1).unsqueeze(1)
    # # save_image(precise_depth/100, 'precise_depth.png')
    # matrix = torch.eye(4).to(depth.device)
    # matrix[0, 0] = matrix[0, 0] / self.downsample
    # matrix[1, 1] = matrix[1, 1] / self.downsample
    # projection = matrix @ final_lidar2img
    # unprojection = torch.inverse(projection.cpu()).to(depth.device)
    # # mask chosen in [segmentation, rangeview_mask, bbox_Mask]
    # complete_points = self.unproject_points(precise_depth, unprojection, rangeview_mask.bool().detach(), crop_top_rate=0.4)
    # # visualization in LIDAR to check if correct
    # self.draw_pts_completion(img_metas, points, complete_points, gt_bboxes_3d, gt_labels_3d)
    # points = [torch.cat([points[i], complete_points[i]], dim=0) for i in range(B)]