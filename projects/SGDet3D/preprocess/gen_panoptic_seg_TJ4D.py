from nuscenes.nuscenes import NuScenes, NuScenesExplorer
import os
import numpy as np
import os
import numpy as np
from PIL import Image, ImageOps
import torch
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from torchvision.utils import save_image
# class_names = ['Pedestrian', 'Cyclist', 'Car'] # VoD
# class_names = ['Pedestrian', 'Cyclist', 'Car','Truck'] # TJ4D

def initialization(save_panoptic_masks_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))

    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
    print("detectron2:", detectron2.__version__)

    ## COCO Label (-1) https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    predictor = DefaultPredictor(cfg)
    save_panoptic_masks_dir = save_panoptic_masks_dir        

    # create dir if not exists
    if not os.path.exists(save_panoptic_masks_dir):
        os.makedirs(save_panoptic_masks_dir)
    return predictor, device

def get_id_from_category_id(segments_info, require_cat):
    '''Returns a list of all output ids to filter. Output Ids 0-8 are moving objects in COCO'
    Input:
        segments_info (dict) - output of the panoptic segmentation model that contains meta information about all the classes
    '''
    output_ids = []
    for segment_info_dict in segments_info:
        if segment_info_dict['category_id'] in require_cat:
            if segment_info_dict['id'] not in output_ids:
                output_ids.append(segment_info_dict['id'])
    return torch.from_numpy(np.asarray(output_ids))

def read_image(file_name, format=None):
    '''Function to read an image
    Source: https://github.com/longyunf/rc-pda/blob/de993f9ff21357af64308e42c57197e8c7307d89/scripts/semantic_seg.py#L59'''
    
    image = Image.open(file_name)

    # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format == "BGR":
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    if format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)
    return image

def read_txt_to_list(file_path):
    lines_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            lines_list.append(line.strip())
    return lines_list

if __name__ == '__main__':
    
    datasets = 'TJ4D' # NOTE: cyclist is bicycle with person, thus here is difference
    if datasets =='VoD': require_cat = [0, 1, 2] # person bicycle car
    if datasets =='TJ4D': require_cat = [0, 1, 2, 6, 7, 8] # person bicycle car bus train truck
    save_panoptic_masks_dir = 'data/TJ4D/segmentation'
    root_path='data/TJ4D/training/image_2'
    ImageSets = 'data/TJ4D/ImageSets/trainval.txt'
    ImageSets = read_txt_to_list(ImageSets)
    predictor, device = initialization(save_panoptic_masks_dir)

    # ImageSets = ['160125']
    with torch.no_grad():
        for image_idx in range(len(ImageSets)):
            filename = ImageSets[image_idx]
            image_path = os.path.join(root_path, filename + '.png')
            raw_image = read_image(image_path, 'RGB')
            
            # network
            panoptic_seg, segments_info = predictor(raw_image)["panoptic_seg"]
            torch.cuda.synchronize(device)

            # filter out all non-moving classes
            panoptic_seg_cpu = panoptic_seg.to("cpu")
            ids_tensor = get_id_from_category_id(segments_info, require_cat=require_cat)
            panoptic_seg_filtered = torch.zeros_like(panoptic_seg_cpu, dtype=torch.bool)
            for id in ids_tensor:
                panoptic_seg_filtered |= (panoptic_seg_cpu == id)
            panoptic_seg_filtered = panoptic_seg_filtered.numpy()
            
            path_seg = os.path.join(save_panoptic_masks_dir, filename + '.png')
            cv2.imwrite(path_seg, 256*panoptic_seg_filtered.astype(np.float32))  
            # cv2.imwrite('seg.png', 256*panoptic_seg_filtered.astype(np.float32))          
            print('compute segmentation %s %d/%d' % (filename, image_idx, len(ImageSets)))