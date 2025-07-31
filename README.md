# SGDet3D: Semantics and Geometry Fusion for 3D Object Detection Using 4D Radar and Camera

## 🗓️ News
- **2025.02.25** all code released
- **2024.12.01** [early access](https://ieeexplore.ieee.org/document/10783046/)
- **2024.11.23** code v1.0 released
- **2024.11.21** RA-L acceptance


https://github.com/user-attachments/assets/3cf6e384-ea33-47b3-b9c9-fe3dc2b77bba
## 📜 Abstract
 
4D millimeter-wave radar has gained attention as an emerging sensor for autonomous driving in recent years. However, existing 4D radar and camera fusion models often fail to fully exploit complementary information within each modality and lack deep cross-modal interactions. To address these issues, we propose a novel 4D radar and camera fusion method, named SGDet3D, for 3D object detection. Specifically, we first introduce a dual-branch fusion module that employs geometric depth completion and semantic radar PillarNet to comprehensively leverage geometric and semantic information within each modality. Then we introduce an object-oriented attention module that employs localization-aware cross-attention to facilitate deep interactions across modalites by allowing queries in bird’s-eye view (BEV) to attend to interested image tokens. We validate our SGDet3D on the TJ4DRadSet and View-of-Delft (VoD) datasets. Experimental results demonstrate that SGDet3D effectively fuses 4D radar data and camera image and achieves state-of-the-art performance.

## 🛠️ Method
![overview](./docs/all_Figures/Framework.png)
Architecture of our SGDet3D neural network. (a) The feature extraction module extract radar and image features. (b) The dual-branch fusion module fully leverages rich radar geometry for image branch and rich image semantics for radar branch, ultimately lifting the features into the unified BEV space. (c) The object-oriented attention module uses cross-attention to further enhance the featurization of the cross-modal BEV queries by deeply interacting with interested image tokens. (d) The object detection head. Dashed line represents the deep utilization of cross-modal information.

## 🍁 Quantitative Results

![View-of-Delft](./docs/all_Figures/Tab-VoD.png)

![TJ4DRadSet ](./docs/all_Figures/Tab-TJ4D.png)

## 🔥 Getting Started

step 1. Refer to [Install.md](./docs/Guidance/Install.md) to install the environment.

step 2. Refer to [dataset.md](./docs/Guidance/dataset.md) to prepare View-of-delft (VoD) and TJ4DRadSet (TJ4D) datasets.

step 3. Refer to [train_and_eval.md](./docs/Guidance/train_and_eval.md) for training and evaluation.

## 🚀 Model Zoo

We retrained the model and achieved better performance compared to the results reported in the tables of the paper. We provide the checkpoints on View-of-delft (VoD) and TJ4DRadSet datasets, reproduced with the released codebase.

|                           Dataset                            | Backbone | EAA 3D mAP | DC 3D mAP |                        Model Weights                         |
| :----------------------------------------------------------: | :------: | :--------: | :-------: | :----------------------------------------------------------: |
| [View-of-delft](projects/SGDet3D/configs/vod-SGDet3D_det3d_2x4_12e.py) | ResNet50 |   59.75    |   77.42   | [Link](https://github.com/shawnnnkb/SGDet3D/releases/download/weights-and-checkpoints/final_ckpt.zip) |
| [TJ4DRadSet](projects/SGDet3D/configs/TJ4D-SGDet3D_det3d_2x4_12e.py) | ResNet50 |   41.82    |   47.16   | [Link](https://github.com/shawnnnkb/SGDet3D/releases/download/weights-and-checkpoints/final_ckpt.zip) |

### 😙 Acknowledgement

Many thanks to these exceptional open source projects:
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [DFA3D](https://github.com/IDEA-Research/3D-deformable-attention.git)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [VoxFormer](https://github.com/NVlabs/VoxFormer.git)
- [OccFormer](https://github.com/zhangyp15/OccFormer.git)
- [CGFormer](https://github.com/pkqbajng/CGFormer)

As it is not possible to list all the projects of the reference papers. If you find we leave out your repo, please contact us and we'll update the lists.

## ✒️ Citation

If you find our work beneficial for your research, please consider citing our paper and give us a star. If you encounter any issues, please contact shawnnnkb@zju.edu.cn.
```
@ARTICLE{SGDet3D,
  author={Bai, Xiaokai and Yu, Zhu and Zheng, Lianqing and Zhang, Xiaohan and Zhou, Zili and Zhang, Xue and Wang, Fang and Bai, Jie and Shen, Hui-Liang},
  journal={IEEE Robotics and Automation Letters}, 
  title={SGDet3D: Semantics and Geometry Fusion for 3D Object Detection Using 4D Radar and Camera}, 
  year={2024}}
```
## 🐸 Visualization Results

![View-of-Delft](./docs/all_Figures/Visualization.png)

Some visualization results on the VoD validation set (first row) and TJ4DRadSet test set (second row). Each figure corresponds to a frame of data containing an image and radar points (gray), with the red triangle marking the ego-vehicle position. Orange and yellow boxes represent ground-truths in the perspective and bird’s-eye views, respectively. Green and blue boxes indicate predicted bounding boxes from SGDet3D, and the bottom left shows BEV feature map visualizations. Figures (a), (b), and (c) demonstrate the detection performance of SGDet3D on cars, cyclists, and pedestrians of VoD, respectively. Figures (d), (e), and (f) illustrate the robustness of SGDet3D in the complex environments of the TJ4DRadSet, such as low-light nighttime conditions and out-of-focus scenarios. Better zoom in for details.
