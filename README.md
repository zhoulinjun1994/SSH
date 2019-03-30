# SSH: Single Stage Headless Face Detector

## Introduction
This repository includes the code for training and evaluating the *SSH* face detector introduced in our [**ICCV 2017 paper**](https://arxiv.org/abs/1708.03979).
We improve the original algorithm and support large image. For original code, please refer to https://github.com/mahyarnajibi/SSH.

### Citing
If you find *SSH* useful in your research please consider citing:
```
@inproceedings{najibi2017ssh,
title={{SSH}: Single Stage Headless Face Detector},
author={Najibi, Mahyar and Samangouei, Pouya and Chellappa, Rama and Davis, Larry},
booktitle={The IEEE International Conference on Computer Vision (ICCV)},
year={2017}
}
```
### Installation
1. Clone the repository:
```
git clone --recursive https://github.com/zhoulinjun1994/SSH.git
```

2. Install [cuDNN](https://developer.nvidia.com/cudnn) and [NCCL](https://github.com/NVIDIA/nccl) (used for multi-GPU training).

3. Caffe and pycaffe: You need to compile the ```caffe-ssh``` repository which is a  Caffe fork compatible with *SSH*. Caffe should be built with *cuDNN*, *NCCL*, and *python layer support* (set by default in ```Makefile.config.example```). You also need to ```make pycaffe```.

4. Install python requirements:
```
pip install -r requirements.txt
```

5. Run ```make``` in the ```lib``` directory:
```
cd lib
make
```

### Training a model
For training on the *WIDER* dataset, you need to download the [WIDER face training images](https://drive.google.com/file/d/0B6eKvaijfFUDQUUwd21EckhUbWs/view?usp=sharing) and the [face annotations](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip) from the [dataset website](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/). These files should be copied into ```data/datasets/wider/``` (you can create symbolic links if you prefer to store the actual data somewhere else).

You also need to download the pre-trained *VGG-16*  ImageNet model. The following script downloads the model into the default directory:
```
bash scripts/download_imgnet_model.sh
```

Before starting to train  you should have a directory structure as follows:
 ```
data
   |--datasets
         |--wider
             |--WIDER_train/
             |--wider_face_split/
   |--imagenet_models
         |--VGG16.caffemodel
```
For training with the default parameters, you can call the ```main_train``` module with a list of GPU ids. As an example:
```
python main_train.py --gpus 0,1,2,3
```
For a list of all possible options run ```python main_train.py --help```.

Please note that the default training parameters (*e.g.* number of iterations, stepsize, and learning rate) are set for training
on 4 GPUs as described in the paper. 

All *SSH* default settings and configurations (saved in ```SSH/configs/default_config.yml```) can be overwritten by passing an external configuration file to the module (```--cfg [path-to-config-file]```. See ```SSH/configs``` for example config files).

By default, the models are saved into the ```output/[EXP_DIR]/[db_name]/``` folder (```EXP_DIR``` is set to ```ssh``` by default and can be changed through the configuration files,
and ```db_name``` would be ```wider_train``` in this case).


### Testing
For testing the model, you need to put the image in folder data/demo/

Then, testing the model with:

```
python demo.py
```

The program will give the execution time and the final face number.
For a test image and its corresponding result, we give a test.jpg under data/demo/.
