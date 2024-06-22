# DLIO reference implementation ML workloads

Read and references will be updated soon.

## Setting up python environments

DLIO and actual workloads may run on different python environments. The scripts are provided for Polaris@ALCF. Please modify accordingly for different supercomputers. 

* DLIO environment
```bash
source setup_dlio_env.sh
```
* ML workloads environment
```bash
source setup_ml_env.sh
```
## Real workloads
- UNet3D 
  This is for medical image segmentation from the reference implementation

- ResNet50
  We have two implementations, one with PyTorch [./resnet50/resnet50_hvd.py](./resnet50/resnet50_hvd.py), one with TensorFlow [./resnet50/resnet50_hvd.py](./resnet50_tf/resnet50_hvd.py)

- CosmoFlow 
  This is a workload adopted from Nvidia's submission of MLPerf HPC.
  
- Deepcam
  PyTorch implementation for the climate segmentation benchmark, based on the
  Exascale Deep Learning for Climate Analytics codebase here:
  https://github.com/azrael417/ClimDeepLearn, and the paper:
  https://arxiv.org/abs/1810.01993

