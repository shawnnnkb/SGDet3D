import torch
import torch.nn as nn
import torch.nn.functional as F
def perspective(matrix, vector): 
    """ Applies perspective projection to a vector using projection matrix from lidar to image coordinates"""
    vector = vector.unsqueeze(-1)

    homogenous = torch.matmul(matrix[..., :-1], vector) + matrix[..., [-1]]
    homogenous = homogenous.squeeze(-1)
    return homogenous[..., :-1] / homogenous[..., [-1]]

def make_grid(grid_size, grid_offset, grid_res, grid_z_min, grid_z_max): 
    """ Constructs an array representing the corners of an orthographic grid in Lidar cordinates"""
    depth, width = grid_size
    xoff, yoff, zoff = grid_offset

    # NOTE: difference between torch.range() and torch.arange()
    xcoords = torch.arange(0., depth+grid_res, grid_res) + xoff
    ycoords = torch.arange(width, 0-grid_res, -grid_res) + yoff
    zcoords = torch.arange(grid_z_max, grid_z_min-grid_res, -grid_res)
    zcoords = F.pad(zcoords.view(-1, 1, 1, 1), [2, 0]) # pad 2 zero in left

    xx, yy = torch.meshgrid(xcoords, ycoords)
    xxyy = torch.stack([xx, yy, torch.full_like(xx, zoff)], dim=-1).unsqueeze(0)
    xxyyzz = xxyy + zcoords
    corners = xxyyzz.unsqueeze(0)
    return corners

class Attention_Block(nn.Module):
    def __init__(self, c):
        super(Attention_Block, self).__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c,c,kernel_size=1,stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.att(x)

class OFT(nn.Module):

    def __init__(self, channels, height_channels, scale=1): 
        super().__init__()
        # self.conv3d = nn.Conv2d((len(y_corners)-1) * channels, channels,1)
        self.conv3d = nn.Linear(height_channels, channels)  ##此处要根据实际改
        self.scale = scale
        self.EPSILON = 1e-6

    def forward(self, features, calib, corners_radar):
        # grid  corners  norm_corners  bbox_corners DO NOT HAVE grad
        # Expand the grid in the y dimension
        assert features.shape[0] == calib.shape[0]
        B = calib.shape[0]
        corners = corners_radar.repeat(B, 1, 1, 1, 1)

        # Project grid corners to image plane and normalize to [-1, 1] using calib
        img_corners = perspective(calib.view(-1, 1, 1, 1, 3, 4), corners)

        # Normalize to [-1, 1], norm_corners:  [B, Z+1, H+1, W+1, 2]
        img_height, img_width = features.size()[2:]  # size of feature map
        img_size = corners.new([img_width, img_height]) / self.scale  # note change hw size
        norm_corners = (2 * img_corners / img_size - 1).clamp(-1, 1)  # 2*[0, 1]-1--->[-1, 1]

        # Get top-left and bottom-right coordinates of voxel bounding boxes
        bbox_corners = torch.cat([ 
            torch.min(norm_corners[:, :-1, :-1, :-1], norm_corners[:, :-1, 1:, :-1]),
            torch.max(norm_corners[:, 1:, 1:, 1:], norm_corners[:, 1:, :-1, 1:])], dim=-1)
        batch, _, depth, width, _ = bbox_corners.size() # B, Z, H, W, 4
        bbox_corners = bbox_corners.flatten(2, 3)

        # Compute the area of each bounding box # if projection outside image, area is 0
        area = ((bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1) * img_height * img_width * 0.25 + self.EPSILON).unsqueeze(1)
        visible = (area > self.EPSILON)  
        
        # Sample integral image at bounding box locations
        integral_img = integral_image(features)
        top_lefts = F.grid_sample(integral_img, bbox_corners[..., [0, 1]], align_corners=False)
        btm_right = F.grid_sample(integral_img, bbox_corners[..., [2, 3]], align_corners=False)
        top_right = F.grid_sample(integral_img, bbox_corners[..., [2, 1]], align_corners=False)
        btm_lefts = F.grid_sample(integral_img, bbox_corners[..., [0, 3]], align_corners=False)

        # Compute voxel features (ignore features which are not visible)
        vox_feats = (top_lefts + btm_right - top_right - btm_lefts) / area
        vox_feats = vox_feats * visible.float()
        vox_feats = vox_feats.permute(0, 3, 1, 2).flatten(0, 1).flatten(1, 2)

        # Flatten to orthographic feature map
        ortho_feats = self.conv3d(vox_feats).view(batch, depth, width, -1)
        ortho_feats = F.relu(ortho_feats.permute(0, 3, 1, 2), inplace=True)
        ortho_feats = ortho_feats.permute(0, 1, 3, 2)
        # Block gradients to pixels which are not visible in the image

        return ortho_feats # (B, C, H, W)


def integral_image(features):
    return torch.cumsum(torch.cumsum(features, dim=-1), dim=-2)


class OftNet(nn.Module):
    def __init__(self, img_bev_harf=True, grid_size=(69.12, 79.36), grid_offset=(0,-39.68, 0), grid_res=0.32, grid_z_min=-3, grid_z_max=2.76, imc=256):
        super().__init__()
        corners_radar = make_grid(grid_size, grid_offset, grid_res, grid_z_min, grid_z_max)
        self.register_buffer('corners_radar', corners_radar)  # (1, 18, 216, 248, 3)
        # Orthographic feature transforms
        height_channels = int((grid_z_max-grid_z_min)/grid_res*imc)
        self.oft04 = OFT(imc, height_channels, 1 / 4.)
        self.oft08 = OFT(imc, height_channels, 1 / 8.)
        self.oft16 = OFT(imc, height_channels, 1 / 16.)
        self.oft32 = OFT(imc, height_channels, 1 / 32.)
        self.oft64 = OFT(imc, height_channels, 1 / 64.)
        
        if img_bev_harf:
            self.bevencoder = nn.Sequential(
                nn.Conv2d(imc*5, imc, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(imc),
                nn.ReLU(inplace=True),
                nn.Conv2d(imc, imc, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(imc),
                nn.ReLU(inplace=True)
            )
        else:
            self.bevencoder = nn.Sequential(
                nn.Conv2d(imc*5, imc, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(imc),
                nn.ReLU(inplace=True),
                nn.Conv2d(imc, imc, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(imc),
                nn.ReLU(inplace=True)
            )
        self.attention_block = Attention_Block(imc)

    def forward(self, img_feats, calib):
        # Normalize by mean and std-dev
        # Run frontend network
        feats04, feats08, feats16, feats32, feats64 = img_feats
        # Apply lateral layers to convert image features to common feature size
        # Apply OFT and sum or cat
        ortho04 = self.oft04(feats04, calib, self.corners_radar)
        ortho08 = self.oft08(feats08, calib, self.corners_radar)
        ortho16 = self.oft16(feats16, calib, self.corners_radar)
        ortho32 = self.oft32(feats32, calib, self.corners_radar)
        ortho64 = self.oft64(feats64, calib, self.corners_radar)
        # Apply attention_block
        ortho04 = self.attention_block(ortho04)
        ortho08 = self.attention_block(ortho08)
        ortho16 = self.attention_block(ortho16)
        ortho32 = self.attention_block(ortho32)
        ortho64 = self.attention_block(ortho64)
        # cat multi level features
        ortho = torch.cat([ortho04, ortho08, ortho16, ortho32, ortho64], dim=1)
        ortho = self.bevencoder(ortho)
        #-------------------ablation-------------------
        # ortho = ortho4+ortho8+ortho16+ortho32+ortho64
        
        return ortho # (B, C, H, W)

if __name__ == '__main__':
    ortho1 = torch.randn((8,256,218,232))
    atten = Attention_Block(256)
    x1 = atten(ortho1)
    print(x1.shape)

    #img_feats = (torch.randn((8,256,240,960), requires_grad=True), torch.randn((8,256,120,480),requires_grad=True),torch.randn((8,256,60,240),requires_grad=True),torch.randn((8,256,30,120),requires_grad=True),torch.randn((8,256,15,60),requires_grad=True))
    # calib = torch.tensor([[0, -1, 0, 0.003],
    #                       [0, 0, -1, 1.334],
    #                       [1, 0, 0, 2.875]]).repeat(8,1,1)
    #
    # BEVencoder = OftNet()
    # out_bev = BEVencoder(img_feats,calib)
    # print(out_bev.shape)

    # img_corners = make_grid((69.12,79.36),(0,-39.68, 0),0.32,-3,2.76)
    # print(img_corners.shape)
    # img_size = img_corners.new([1280, 960])
    # norm_corners = (2 * img_corners / img_size - 1).clamp(-1, 1)
    #
    # bbox_corners = torch.cat([
    #     torch.min(norm_corners[:, :-1, :-1, :-1],  # 这里索引的是前四维，先取小uv再取大
    #               norm_corners[:, :-1, 1:, :-1]),
    #     torch.max(norm_corners[:, 1:, 1:, 1:],
    #               norm_corners[:, 1:, :-1, 1:])
    # ], dim=-1)
    # bbox_corners = bbox_corners.flatten(2, 3)
    # print(bbox_corners.shape)
    # area = ((bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1) \
    #         * 360 * 1080 * 0.25*0.125*0.5 + EPSILON).unsqueeze(1)
    # print(area)
    # print(area.shape)