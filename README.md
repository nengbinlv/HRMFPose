# Hybrid Representation and Multi-Feature Fusion for Robust 6D Pose Estimation in Industrial Assembly
# Overview
we propose a novel 6D pose estimation method via hybrid representations and multi-feature fusion for complex industrial assembly.In the first stage, an initial pose estimation module based on multi-task learning is proposed. This module predicts hybrid representations, including sparse keypoint heatmaps, keypoint relational vectors, keypoint visibilities, semantic edges, and semantic masks of parts. RANSAC+PnP is employed to obtain initial pose estimates. In the second stage, a multi-feature fusion pose optimization method is proposed, which combines learned high-dimensional semantic features with extracted general features and performs iterative optimization to obtain accurate pose results. For multi-object pose estimation in assembly scenarios, a structural constraint strategy is employed to correct the poses.
![Proposed method](https://github.com/nengbinlv/HRMFPose/blob/main/assets/framework.png)
# Environment Installation
## python

## c++
