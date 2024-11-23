import torch
from collections import OrderedDict
checkpoint = torch.load("/mnt/disk8T/zhenglianqing/4-dradar_mmdet3d4/checkpoints/Radarpillarnet_0.32_epoch_79.pth")
# for key, value in checkpoint['state_dict'].items():
#     print(key)

if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif 'model' in checkpoint:
    state_dict = checkpoint['model']
else:
    state_dict = checkpoint
if list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k[7:]: v for k, v in state_dict.items()}
ckpt = state_dict
new_ckpt = OrderedDict()
for k, v in ckpt.items():
    if k.startswith('backbone'):
        new_v = v
        new_k = k.replace('backbone.', 'pts_backbone.')
    elif k.startswith('neck'):
        new_v = v
        new_k = k.replace('neck.', 'pts_neck.')
    elif k.startswith('voxel_encoder'):
        new_v = v
        new_k = k.replace('voxel_encoder.', 'pts_voxel_encoder.')
    else:
        continue
    new_ckpt[new_k] = new_v

for key, value in new_ckpt['state_dict'].items():
    print(key)
