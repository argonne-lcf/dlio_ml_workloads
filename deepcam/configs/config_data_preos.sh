#!/bin/bash

# data directory
export DATADIR=/lustre/fsw/mlperft/data/deepcam/numpy
export DATADIR_FUSED=/lustre/fsw/mlperft/data/deepcam/numpy_fused

# throughput data directories
export DATADIR_SMALL=/lustre/fsw/mlperft/data/deepcam/numpy_small

# JET
export JET_DIR="/project/mlperft/common/jet2"
export JET_UPLOAD="jet logs upload output.zip"
export JET_CREATE="jet logs create output.zip --fill-gpu --fill-cpu --fill-system --fill-libraries --data user=${USER} --data workload.maintainers[]=${USER} --data type=workload --data workload.type=custom --data origin=${JET_ORIGIN:-"mlperf-manual"} --data workload.spec.script=run_and_time.sh "

# Readme prefix for JSON file
export README_PREFIX="https://gitlab-master.nvidia.com/dl/mlperf/optimized-hpc/-/blob/main"