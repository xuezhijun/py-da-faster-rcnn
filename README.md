# Pytorch implementation of domain adaptive Faster RCNN serials

## Requirements

pytorch 0.4.0 and torchvision 0.2.0

To install requirements, run:
```bash
git clone https://github.com/xuezhijun/py-da-faster-rcnn.git
cd py-da-faster-rcnn
pip install -r requirements.txt
```

Choose the right `-arch` in `make.sh` file to compile the CUDA code, for example, sm_70 for V100.
```bash
cd lib
sh make.sh
```

## Datasets and pre-trained models

Download pretrained models (vgg16_caffe.pth、res50_caffe.pth、res101_caffe.pth) and datasets (Sim10k、Cityscapes、SYNTHIA-RAND-CITYSCAPES、Foggy-Cityscapes、Clipart、WaterColor、VOC 2007、VOC2012 etc).
Dataset and pretrained models' path can be reset in <lib/model/utils/config.py>.

## Training

To train the models with CUDA, run:

```bash
CUDA_VISIBLE_DEVICES=7 python tools/trainval_net.py --cuda --net vgg16 --dataset sim10k
```

```bash
CUDA_VISIBLE_DEVICES=7 python tools/trainval_net_<x>.py --cuda --net vgg16 --dataset sim10k --dataset_t cityscape_car
```

```bash
CUDA_VISIBLE_DEVICES=7 python tools/trainval_net.py --cuda --net vgg16 --dataset sim10k --dataset_t cityscape_car --r --load_name models/vgg16/sim10k/frcnn.pth --start_epoch 5 --max_epochs 10
```

```bash
CUDA_VISIBLE_DEVICES=7 python tools/demo_for_da.py --cuda --net vgg16 --dataset_t cityscape_car --da_method DA --load_name models/vgg16/sim10k/frcnn.pth
```

>📋 Currently support only batch_size = 1. batch_size > 1 may cause some errors.

## Evaluation

To evaluate the models, run:

```bash
CUDA_VISIBLE_DEVICES=7 python tools/test_net.py --cuda --net vgg16 --dataset cityscape_car --r --load_name models/vgg16/sim10k/frcnn.pth
```

```bash
python tools/iterative_test.py --gpu_id 0 --start_epoch 1 --max_epochs 10 --test_script tools/test_net_<x>.py --net vgg16 --dataset cityscape_car --load_name models/vgg16/frcnn.pth
```

## Paper list

Domain Adaptive Faster R-CNN for Object Detection in the Wild, in *CVPR*, 2018.

Strong-Weak Distribution Alignment for Adaptive Object Detection, in *CVPR*, 2019.

SCL: Towards Accurate Domain Adaptive Object Detection via Gradient Detach Based Stacked Complementary Losses, 2019.

Cross-domain Detection via Graph-induced Prototype Alignment, in *CVPR*, Virtual, 2020.

Exploring Categorical Regularization for Domain Adaptive Object Detection," in *CVPR*, Virtual, 2020.

More to do.
