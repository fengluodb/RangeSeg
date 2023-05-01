# LENet
> This is the official implementation of **LENet: Lightweight And Efficient LiDAR Semantic Segmentation Using Multi-Scale Convolution Attention**[[Paper](https://arxiv.org/pdf/2301.04275.pdf)]. [![arXiv](https://img.shields.io/badge/arxiv-2202.13377-b31b1b.svg)](https://arxiv.org/abs/2301.04275) 
## Demo
<p align="center"> <img src="assert/Network.png" width="100%"></a> </p>

## Environment
```sh
# clone this repo
git clone https://github.com/fengluodb/LENet.git

# create a conda env with
conda env create -f environment.yaml
conda activate LENet
```

## Datasets Prepartion
### SemanticKITTI
Download the SemanticKIITI dataset from [here](http://www.semantic-kitti.org/dataset.html#download). 
```
dataset
└── SemanticKITTI
    └── sequences
        ├── 00
        ├── ...
        └── 21
```


### SemanticPOSS
Download the SemanticPOSS dataset from [here](http://www.poss.pku.edu.cn./semanticposs.html). Unzip and arrange it as follows. 
```
dataset
└── SemanticPOSS
    └── sequences
        ├── 00
        ├── ...
        └── 05
```

## Training

### SemanticKITTI
To train a network (from scratch):
```sh
python train.py -d DATAROOT -ac config/arch/LENet.yaml -dc config/labels/semantic-kitti.yaml -l logs/LENet-KITTI
```

To train a network (from pretrained model):
```sh
python train.py -d DATAROOT -ac config/arch/LENet.yaml -dc config/labels/semantic-kitti.yaml -l logs/LENet-KITTI -p "logs/LENet-KITTI/TIMESTAMP" 
```

### SemanticPOSS
To train a network (from scratch):
```sh
python train.py -d DATAROOT -ac config/arch/LENet_poss.yaml -dc config/labels/semantic-poss.yaml -l logs/LENet-POSS
```

To train a network (from pretrained model):
```sh
python train.py -d DATAROOT -ac config/arch/LENet_poss.yaml -dc config/labels/semantic-poss.yaml -l logs/LENet-POSS -p "logs/LENet-POSS/TIMESTAMP""
```

## Inference

### SemanticKITTI
```sh
python infer.py -d DATAROOT -m "logs/LENet-KITTI/TIMESTAMP" -l /path/for/predictions -s valid/test
```

### SemanticPOSS
```sh
python infer.py -d DATAROOT -m "logs/LENet-POSS/TIMESTAMP" -l /path/for/predictions -s valid
```

## Evalution

### SemanticKITTI
```sh
python evaluate.py -d DATAROOT -p /path/for/predictions -dc config/labels/semantic-kitti.yaml
```

### SemanticPOSS
```sh
python evaluate.py -d DATAROOT -p /path/for/predictions -dc config/labels/semantic-poss.yaml
```

## Pretrained Models and Predictions

| dataset | test mIoU |  Download |
|---------------|:----:|:-----------:|
| [SemanticKITTI(single)](config/arch/LENet.yaml) | 64.5 | [Model Weight And Predictions](https://drive.google.com/drive/folders/1ejoInYl8BVzg3t69_ig4tDUYstaz--Ns?usp=sharing) |
| [SemanticKITTI(multi)](config/arch/LENet.yaml) | 53.0 | [Model Weight And Predictions](https://drive.google.com/drive/folders/1OfktGL85mFmdRALBb-_Zpc8VSmjXJVYU?usp=sharing) |
| [SemanticPOSS](config/arch/LENet_poss.yaml) | 53.8 | [Model Weight And Predictions](https://drive.google.com/drive/folders/1oECv2GRCXZ1RIQVVum-mRwZbod8pxVA8) |

## Acknowlegment

This repo is built based on [MotionSeg3D](https://github.com/haomo-ai/MotionSeg3D), [FIDNet](https://github.com/placeforyiming/IROS21-FIDNet-SemanticKITTI) and [CENet](https://github.com/huixiancheng/CENet). Thanks the contributors of these repos!
