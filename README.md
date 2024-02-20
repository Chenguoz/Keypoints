# Unsupervised Distribution-aware Keypoints Generation from 3D Point Clouds

## Description
This repository contains the code for our paper: **Unsupervised Distribution-aware Keypoints Generation from 3D Point Clouds**.
> [**Unsupervised Distribution-aware Keypoints Generation from 3D Point Clouds**](https://doi.org/10.1016/j.neunet.2024.106158),            
> Yiqi Wu, Xingye Chen, Xuan Huang, Kelin Song, Dejun Zhang            
> [Bibetex](https://github.com/Chenguoz/Keypoints#citation)

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


## Citation

```bibtex
@article{wu2024unsupervised,
  title={Unsupervised distribution-aware keypoints generation from 3D point clouds},
  author={Wu, Yiqi and Chen, Xingye and Huang, Xuan and Song, Kelin and Zhang, Dejun},
  journal={Neural Networks},
  pages={106158},
  year={2024},
  publisher={Elsevier}
}
```

## Acknowledgment

Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[3DStructurePoints](https://github.com/NolenChen/3DStructurePoints),
[Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)
