# Research

## Comparing Detectron2 with other libraries
**There are only the configurations which were used in notebooks*

Name | Detectron2 | Mediapipe | Alphapose
---|:----:|:----:|:----:
GitHub | [detectron2](https://github.com/facebookresearch/detectron2) | [mediapipe](https://github.com/google/mediapipe) | [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)
Description |  Object detection platform | Estimates the full 3D body pose. Optimized for on-device, real-time fitness applications | Multi-person pose estimator
Architecture | Mask R-CNN (Resnet101) | Mobilenet V2 | Resnet
Back | Pytorch | TFlite | Pytorch
Inference time | - | lite: 44fps cpu, 112fps gpu | -
AP @[0.5-0.95] | 78 | - | 73
Keypoints | 17 | 33 | 17


### Detectron2 
Detectron2 Tutorial - working with outputs [<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/drive/1xDM-XmZdxcpC6DdA1ihb6QVaftqA-L6M?usp=sharing)

Detectron2 Tutorial - video inference [<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/drive/1zHuw-xRPl8iIpJIW7mkr_4He8OMDAXer?usp=sharing)

Detectron2 Tutorial - training with a COCO dataset [<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/drive/1ka8tgaa1p0YN4ipCNJCZZ9SohB5fXcPE?usp=sharing)

* Image

![img](images/ex1.png)
  
* Video

![img](images/dtr.gif)

### AlphaPose
[<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/drive/1udurAuDp3TTFflbxTfRvf57YMdzfjKkh?usp=sharing)

* Image

![img](images/example_.jpg)
  
* Video

![img](images/alpha.gif)

### Google Mediapipe
[<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/drive/126vr1wK4FgH-AggGmONNFrz6-HbA-xXM?usp=sharing)

* Image

![img](images/google.png)
  
* Video

![img](images/google2.gif)

[*installation on EC2*](https://dev.classmethod.jp/articles/mediapipe-install-on-aws-ec2-with-gpu-english/)

## Get 3d from 2d video

### Detectron2 + VideoPose3d
[<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/drive/18bn88ALDOzbeuuJ1Rbj5_yDDroCuJP7X?usp=sharing)
![img](images/output.gif)

### Metrabs
[<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/drive/1SP2fLqvtT1wyAHBIKC2pPUKq83nseHW6?usp=sharing)
![img](images/metrab.png)