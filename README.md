# DLIO reference implementation ML workloads

Read and references will be updated soon.

## Setting up python environments

DLIO and actual workloads may run on different python environments. The scripts are provided for Polaris@ALCF. Please modify accordingly for different supercomputers. 

* DLIO python environment
  ```bash
  source setup_dlio_env.sh
  ```
* ML workloads python environment
  ```bash
  source setup_ml_env.sh
  ```
## Real workloads
- UNet3D

  This is for medical image segmentation from the reference implementation

- ResNet50

  We have implementations for both TensorFlow and PyTorch 
   * TensorFlow: [./resnet50_tf/resnet50_hvd.py](./resnet50_tf/resnet50_hvd.py)
   * PyTorch: [./resnet50_pt/resnet50_hvd.py](./resnet50_pt/resnet50_hvd.py) and [./resnet50_pt/resnet50_ddp.py](./resnet50_pt/resnet50_ddp.py), 

- CosmoFlow 

  This is a workload adopted from Nvidia's submission of MLPerf HPC.
  
- Deepcam

  PyTorch implementation for the climate segmentation benchmark, based on the
  Exascale Deep Learning for Climate Analytics codebase here:
  https://github.com/azrael417/ClimDeepLearn, and the paper:
  https://arxiv.org/abs/1810.01993

