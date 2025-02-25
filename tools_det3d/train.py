# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import copy, argparse, os, time
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import mmcv, shutil, torch, warnings
torch.autograd.set_detect_anomaly(True)
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp
from collections import OrderedDict
from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version
import wandb
from mmcv.runner.dist_utils import master_only

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default='projects/RCdistill/configs/vod-RCdistill_teacher_pretrain_2x4_12e.py', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args

@master_only
def mmdet3d_wandb_init(cfg, project):
    # project=project
    wandb.init(project='JUST_DEBUG', entity="shawnnnkb", name='test0', config=cfg)

def load_pretrained_model(model, check_point_path, mapping_list):
    if check_point_path is None:
        print('check_point_path is in valid, fail to load pretrained model')
        return model
    checkpoint = torch.load(check_point_path, map_location='cpu')
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
        for old_key, new_key in mapping_list.items():
            if k.startswith(old_key):
                new_k = k.replace(old_key, new_key)
                new_ckpt[new_k] = v
                break
    if list(mapping_list.values())[0] == 'img_backbone_extra':
        new_ckpt.pop('img_backbone_extra.stem.stem_1/conv.weight')
    missing_keys, unexpected_keys = model.load_state_dict(new_ckpt, strict=False)
    # if len(missing_keys) > 0: print("Missing keys:", missing_keys)
    # if len(unexpected_keys) > 0: print("Unexpected keys:", unexpected_keys)
    return model

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # wandb init after runner set up main process  
    project = args.config.split('/')[-1].split('.')[0]
    if 'tj4d' in project.lower()[:5]:
        src_path = 'tools_det3d/eval_tools/TJ4D-eval.py'
        dst_path = 'mmdet3d/core/evaluation/kitti_utils/eval.py'
        shutil.copy(src_path, dst_path)
        src_path = 'tools_det3d/eval_tools/TJ4D-kitti_dataset.py'
        dst_path = 'mmdet3d/datasets/kitti_dataset.py'
        shutil.copy(src_path, dst_path)
        print('USING EVAL TOOLS OF TJ4D DATASET')
    if 'vod' in project.lower()[:5]:
        src_path = 'tools_det3d/eval_tools/vod-eval.py'
        dst_path = 'mmdet3d/core/evaluation/kitti_utils/eval.py'
        shutil.copy(src_path, dst_path)
        src_path = 'tools_det3d/eval_tools/vod-kitti_dataset.py'
        dst_path = 'mmdet3d/datasets/kitti_dataset.py'
        shutil.copy(src_path, dst_path)
        print('USING EVAL TOOLS OF VOD DATASET')
        
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
        
    if not 'pillar' in project:
        figures_path = os.path.join(cfg.work_dir, 'figures_path')
        os.makedirs(figures_path, exist_ok=True)
        cfg.model.update(meta_info = {'figures_path':figures_path, 'project_name': project})
        
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    logger.info(f'Model:\n{model}')
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    
    #################################### pretrained radar for furthur RC pretrain ################################
    if 'load_radar_from' in cfg:
        print('load_radar_from is exists, which means we load pretrained pure radar ')
        print('model, thus need param state_dict mapping, %s'%(cfg.load_radar_from))
        mapping_list = {
            'voxel_encoder': 'pts_voxel_encoder',
            'middle_encoder': 'pts_middle_encoder',
            'backbone': 'pts_backbone',
            'neck': 'pts_neck'}
        model = load_pretrained_model(model, cfg.load_radar_from, mapping_list)
    #################################### pretrained lidar for furthur RC pretrain ################################
    if 'load_lidar_from' in cfg:
        print('load_lidar_from is exists, which means we load pretrained pure lidar ')
        print('model, thus need param state_dict mapping, %s'%(cfg.load_radar_from))
        mapping_list = {
            'voxel_encoder': 'lidar_pts_voxel_encoder',
            'middle_encoder': 'lidar_pts_middle_encoder',
            'backbone': 'lidar_pts_backbone',
            'neck': 'lidar_pts_neck'}
        model = load_pretrained_model(model, cfg.load_lidar_from, mapping_list)
        
    # mmdet3d_wandb_init(cfg, project)
    print('MODEL TOTAL PARAMETERS = %d'%(sum(p.numel() for p in model.parameters())))
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)

if __name__ == '__main__':
    main()
