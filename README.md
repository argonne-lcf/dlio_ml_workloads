# DLIO validation for ML workloads

Read and references will be updated soon.

## Setting up python environments

DLIO and actual workloads may run on different python environments

* DLIO envs
```bash
source setup_dlio_env.sh
```
* Environement for running actual workloads
```bash
source setup_ml_env.sh
```

## Real workloads
- ResNet50
  We have two implementations, one with PyTorch [./resnet50/resnet50_hvd.py](./resnet50/resnet50_hvd.py), one with TensorFlow [./resnet50/resnet50_hvd.py](./resnet50/resnet50_hvd.py)
- CosmoFlow 
  This is a workload adopted from Nvidia's submission of MLPerf HPC
- UNet3D 
  This is for medical image segmentation from the reference implementation
- deepcam
  PyTorch implementation for the climate segmentation benchmark, based on the
  Exascale Deep Learning for Climate Analytics codebase here:
  https://github.com/azrael417/ClimDeepLearn, and the paper:
  https://arxiv.org/abs/1810.01993

