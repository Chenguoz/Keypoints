# Unsupervised Uniformly Distributed Keypoints Generation for 3D Model

## Description
This repository contains the code for our paper: Unsupervised Uniformly Distributed Keypoints Generation for 3D Model.

<div align="center">
<img src="https://github.com/Chenguoz/Unsupervised3D/blob/main/images/Pipeline.png" width="70%" height="70%"><br><br>
</div>

<div align="center">
<img src="https://github.com/Chenguoz/Unsupervised3D/blob/main/images/VisualizationResults.png" width="70%" height="70%"><br><br>
</div>


## Environment setup
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install pointnet2_ops_lib/.
```
## Dataset
The training and testing data for correspondence is provided by [KeypointNet](https://github.com/qq456cvb/KeypointNet) and [ShapeNet](https://github.com/antao97/PointCloudDatasets)

## Acknowledgment

Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[3DStructurePoints](https://github.com/NolenChen/3DStructurePoints),
[Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)
