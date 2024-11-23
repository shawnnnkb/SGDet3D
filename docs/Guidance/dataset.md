## View-of-delft (VoD)

Download the [View-of-delft](https://github.com/tudelft-iv/view-of-delft-dataset). Then we generate perspective view foreground segmentation results from detectron2 as one of the ground-truths. All our ablation experiments are conducted on the VoD dataset.

### preparation
```bash
ln -s /your_path/view_of_delft_PUBLIC/ ./data/VoD
python projects/SGDet3D/preprocess/gen_panoptic_seg_vod.py
python projects/SGDet3D/preprocess/png2npy_vod.py
python tools_det3d/create_data_VODradar.py # creat radar_5frames data as radar data
python tools_det3d/create_data_VODlidar.py # creat lidar data, for lidar detection
```

### Folder structure

The data is organized in the following format:

```
View-of-Delft-Dataset (root)
    ├── lidar (VoD dataset where velodyne contains the LiDAR point clouds)
    │   │── ImageSets
    │   │── training
    │   │   ├──calib & velodyne & image_2 & label_2
    │   │── testing
    │       ├──calib & velodyne & image_2
    | 
    ├── radar (VoD dataset where velodyne contains the radar point clouds)
    │   │── ImageSets
    │   │── training
    │   │   ├──calib & velodyne & image_2 & label_2
    │   │── testing
    │       ├──calib & velodyne & image_2
    | 
    ├── radar_3_scans (VoD dataset where velodyne contains the accumulated radar point clouds of 3 scans)
    │   │── ImageSets
    │   │── training
    │   │   ├──calib & velodyne & image_2 & label_2
    │   │── testing
    │       ├──calib & velodyne & image_2
    |
    ├── radar_5_scans (VoD dataset where velodyne contains the radar point clouds of 5 scans)
        │── ImageSets
        │── training
        │   ├──calib & velodyne & image_2 & label_2
        │── testing
            ├──calib & velodyne & image_2
          
```

## TJ4DRadSet

Download the dataset from [TJ4DRadSet](https://github.com/TJRadarLab/TJ4DRadSet) and prepare the segmentation mask as mentioned above. Since the LiDAR has not been released, we use the radar depth map as the depth supervision information.
### preparation
```bash
ln -s /your_path/TJ4DRadSet_4DRadar/ ./data/TJ4D
python projects/SGDet3D/preprocess/gen_panoptic_seg_TJ4D.py
python projects/SGDet3D/preprocess/png2npy_TJ4D.py
python tools_det3d/create_data_TJ4Dradar.py # creat radar data (1 scan)
```
### Folder Structure

The dataset is organized similarly to KITTI as follows.

```
TJ4DRadSet_4DRadar
    ├── ImageSets
    │   │── train.txt
    |       ...
    │   │── readme.txt
    |
    ├── training
    │   │── calib
    │       ├──000000.txt
    │       ...
    │   │── image_2
    │       ├──000000.png
    │       ...
    │   │── label_2
    │       ├──020000.txt
    │       ...    
    │   │── velodyne
    │       ├──000000.bin
    │       ...  
    ├── Video_Demo
    │   │── seq04.mp4
    │       ...  

TJ4DRadSet_LiDAR
    ├── (On going)
```