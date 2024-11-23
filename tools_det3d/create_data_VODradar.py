# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp

from tools_det3d.data_converter import kitti_converter_VODradar as kitti


def kitti_data_prep(root_path, info_prefix, version, out_dir):
    """Prepare data related to Kitti dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.
    """
    kitti.create_kitti_info_file(root_path, info_prefix)    # train val trainval test all information pkl
    kitti.create_reduced_point_cloud(root_path, info_prefix) # velodyne_reduced in image view

parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('--dataset', default='vod', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/VoD/radar_5frames', 
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/VoD/radar_5frames',
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='vod')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'kitti':
        kitti_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir)
    elif args.dataset == 'vod':
        kitti_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir)
    elif args.dataset == 'TJ4DRadSet_LiDAR':
        kitti_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir)
    elif args.dataset == 'TJ4DRadSet_4DRadar':
        kitti_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir)
    elif args.dataset == 'TJ4DRadSet_4DRadar_filter':
        kitti_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir)