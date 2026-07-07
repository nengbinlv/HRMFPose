# Hybrid Representation and Adaptive Multi-Feature Fusion for Monocular 6D Pose Estimation in Industrial Assembly
The source code in this repository corresponds to our paper submitted to __The Visual Computer__, titled __Hybrid Representation and Adaptive Multi-Feature Fusion for Monocular 6D Pose Estimation in Industrial Assembly__.
# Overview
We propose a novel 6D pose estimation method via hybrid representations and multi-feature fusion for complex industrial assembly.In the first stage, an initial pose estimation module based on multi-task learning is proposed. This module predicts hybrid representations, including sparse keypoint heatmaps, keypoint relational vectors, keypoint visibilities, semantic edges, and semantic masks of parts. RANSAC+PnP is employed to obtain initial pose estimates. In the second stage, a multi-feature fusion pose optimization method is proposed, which combines learned high-dimensional semantic features with extracted general features and performs iterative optimization to obtain accurate pose results. For multi-object pose estimation in assembly scenarios, a structural constraint strategy is employed to correct the poses.
![Proposed method](https://github.com/nengbinlv/HRMFPose/blob/main/assets/framework_2.png)
# Environment Installation
- ## Hardware Environment
windows 11
- ## python
```
conda create --name HRMFPose python=3.8
conda activate HRMFPose
# install the pytorch version compatible with the your cuda version
pip install -r requirements.txt
```
- ## c++
Install Visual Studio 2017
Install Eigen 3, GLEW, GLFW 3, and OpenCV 4
For details, please refer to [3DObjectTracking](https://github.com/DLR-RM/3DObjectTracking/tree/master?tab=readme-ov-file).
# Data Preparation
- ## Industrial Assembly Pose Dataset

Download the [assembly dataset](https://zenodo.org/records/21245467) . Extract it into the ```data``` folder. The file structure is as follows:
```
${PROJECT_ROOT}
 -- data
     -- part_01
         |-- edge_occ
         |-- gtEdge
         |-- mask
         |-- model
         |-- photo_cut
            |-- train
            |-- val
         |--render
            |-- edge_occ
            |-- gtEdge
            |-- mask
            |-- rgb_bg
            |-- pose_final.yml
         |-- gt.yml
         |-- part_01.txt
     -- part_02
         ...
```
```gt.yml``` and ```pose_final.yml``` represent the ground truth poses. ```part_01.txt``` defines the keypoints of the target. ```model``` contains the 3D model. Other folders store the corresponding images.
For information on splitting the training and test sets, please refer to ```dataSplit.py```.
![data1](https://github.com/nengbinlv/HRMFPose/blob/main/assets/data1.png)
- ## Mono6D dataset

The original source of this dataset is [Mono6D](https://isl.sist.chukyo-u.ac.jp/Archives/Mono-6D.zip). We modified it to fit the structure of the proposed method. The structure is consistent with the assembly dataset. Download link: [Mono6D_ours](https://pan.baidu.com/s/14xmeC0hvZp09ajlMMEdmWw). (Extraction code: ```jnsy```).
![data2](https://github.com/nengbinlv/HRMFPose/blob/main/assets/data2.png)
# Train
- ## Train assembly dataset
Run the following script
```
python main_assembly.py --class_type part_02 --batch_size 6 --train True --epochs 150 --eval False
```
- ## Train Mono6D dataset
```
python main_Mono6D.py --class_type Bracket --batch_size 6 --train True --epochs 150 --eval False
```
# Evaluation
- ## Evaluate assembly dataset
```
python main_assembly.py --class_type part_02 --batch_size 1 --train False ----used_epoch 150 --eval True
```
- ## Evaluate Mono6D dataset
```
python main_Mono6D.py --class_type Bracket --batch_size 1 --train False ----used_epoch 150 --eval True
```
For more metric evaluation methods, please refer to ```CalADDmetric.py```.
# Demo
You can run a demo case by executing ```demo.py```.
The pre-trained weights can be downloaded from [weights of assembly dataset](https://pan.baidu.com/s/1BJfG62xOJsX2j2F6zkgDjg) (extraction code: ```a5ca```) and [weights of Mono6D dataset](https://pan.baidu.com/s/1mlvNoznYdam17-JY3rjSig) (extraction code:```47ic```), and placed in the ```weights``` folder.
# Pose Optimization
Pose optimization is implemented by running the ```ObjectTracking.cpp``` file. The input of ```pose_txt``` is the initial pose estimation result, which comes from the above prediction. ```rgbImg_path```, ```edgeImg_path```, and ```maskImg_path``` are the paths of the input image, the predicted semantic edge and the mask image, respectively.
# Multi-object pose correction
After obtaining the individual poses of multiple assembly parts, pose correction is realized by running ```multipleObjectPose.py```.
# Visualization
- Results of assembly dataset
![pose1](https://github.com/nengbinlv/HRMFPose/blob/main/assets/pose1.png)
- Results of Mono6D dataset
![pose2](https://github.com/nengbinlv/HRMFPose/blob/main/assets/pose2.png)
# Acknowledgement
This work is developed on the basis of the following projects. We would like to express our sincere gratitude to the authors for their high-quality open-source contributions.
- [SRT3D](https://github.com/DLR-RM/3DObjectTracking/tree/master/SRT3D)
- [ContourPose](https://github.com/ZJU-IVI/ContourPose)
# Cite
If you find our work useful, please cite us with:
```
@Article{
  author  = {Nengbin Lv, Zhangmao Xu, Yi Feng, Weikai Zeng, Fuzhou Du},
  title   = {Hybrid Representation and Adaptive Multi-Feature Fusion for Monocular 6D Pose Estimation in Industrial Assembly},
  note   = {Submitted to The Visual Computer}
  year    = {2026}
}
```
