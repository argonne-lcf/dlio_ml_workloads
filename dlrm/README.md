# Running DLRM on Polaris

## Environment Setup

Running `0_setup_env.sh` on a compute node (via `qsub -I`) will clone both the `mlcommons/training` and `torchrec` repositories and create a Conda environment with requirements installed. This script will perform the following tasks:

- Load necessary `module`s
- Clone `mlcommons/training` and apply the `mlcommons_training.patch` patch file
- Clone `torchrec` and apply the `torchrec.patch` patch file
- Create a Conda environment with Python 3.7
    - Install `mlcommons/training` and `torchrec` requirements
    - Install DFTracer

> [!NOTE]
> - Python version 3.7 is required per `fbgemm-gpu` and `torchrec` requirements
> - CUDA Toolkit v11 is required per `fbgemm-gpu` and `torchrec` requirements
> - `torchrec==0.3.2` is required per `mlcommons/training` requirements
> - Patching via `git apply` is required for tracing and is done automatically

## Downloading Dataset

The workload uses the Criteo 1TB Click Logs dataset. The dataset should be downloaded manually through the following steps:

- Open [the dataset page](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) and click on the download link
- Click "Download" on WeTransfer's download page
- After the download begins, right-click on the file being downloaded and select "Copy Download Link"
- Then download the dataset on your machine using the following command

```bash
wget --user-agent Mozilla/4.0 '[your big address here]' -O dest_file_name
```

## Splitting Dataset

To streamline the training and overcome the memory requirement to preprocess the dataset, the dataset should be split into smaller parts via `1_split_dataset.sh`. An example usage of the script is the following:

```bash
./1_split_dataset.sh <dataset_dir> <split_dir> [split_line]
```

Running this script will do the following:

- Split the `day_0` log file into smaller parts
    - The default number of lines to split is 5 million lines (`split -l 5000000`)
- Rename log files to `day_0`, `day_1`, ...
- Remove extra parts

> [!NOTE]
> The preprocessing and training require a specific number of log files (for 24 days) and a specific file naming, that's why we only keep the parts `0..23` in the split folder.

## Preprocessing Dataset

The dataset preprocessing is required to change the dataset format from TSV to NPY and can be done by submitting the `2_preprocess_dataset.sh` script via `qsub`. An example submission command is the following:

```bash
qsub -v "DATASET_DIR=/path/to/split_dir,WORKDIR=/path/to/project/root" 2_preprocess_dataset.sh
```

## Training

To launch the training on Polaris, the `3_train.sh` batch script should be submitted with the following variables:

```bash
qsub -v "DATASET_DIR=/path/to/final_dir,WORKDIR=/path/to/project/root" 3_training.sh
```

> [!NOTE]
> Note that this time, the `DATASET_DIR` refers to the location of the preprocessed dataset (after splitting and preprocessing).

Running this script will do the following:

- Set necessary environment variables
- Save environment variables, available GPUs and workload parameters in the output folder
- Run the training via `mpiexec` 
- Convert trace files into analysis-ready Perfetto files

<details open>
    <summary>Key Environment Variables</summary>
    
    `DFTRACER_DATA_DIR`: must refer to the dataset location
    `DFTRACER_ENABLE`: must be `1` to collect traces
    `DFTRACER_LOG_FILE`: must be set
    `LOCAL_WORLD_SIZE`: must be equal to the number of GPUs (in the `launcher.sh` file) on a single node
    `MASTER_ADDR`: crucial for the distributed process group in PyTorch
    `MASTER_PORT`: must be set (a random value is fine)
    `NCCL_COLLNET_ENABLE`: enables CollNet to optimize collective communication
</details>

A sample output can be found in the `dlrm/sample_log.txt` file. 
