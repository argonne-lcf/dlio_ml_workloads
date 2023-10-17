#!/bin/bash

# data directory
source $(dirname ${BASH_SOURCE[0]})/config_DGXH100_common.sh
export DATADIR=${DATADIR_SMALL}

# hyperparameters
export LOCAL_BATCH_SIZE=8
export START_LR=0.00155
export OPTIMIZER="MixedPrecisionLAMB"
export LR_SCHEDULE_TYPE="cosine_annealing"
export LR_T_MAX="9000"
export LR_ETA_MIN="0.0"
export LR_WARMUP_STEPS=0
export LR_WARMUP_FACTOR=1.
export WEIGHT_DECAY=0.01
export BATCHNORM_GROUP_SIZE=1

# data parameters
export SHUFFLE_MODE="global"
export DATA_FORMAT="dali-dummy"
export PRECISION_MODE="amp"
export LOCAL_VALIDATION_BATCH_SIZE=8

# misc args
export MAX_EPOCHS=1
export ADDITIONAL_ARGS="${ADDITIONAL_ARGS} --disable_comm_overlap --enable_graph --enable_odirect --target_iou=1.01"

# system parameters
export DGXNNODES=1
WALLTIME_MINUTES=40
export WALLTIME=$(( 15 + (${NEXP} * ${WALLTIME_MINUTES}) ))
#export SBATCH_NETWORK="sharp"

# dltools
export CONFIG=$(basename ${BASH_SOURCE[0]%.*} | awk '{split($1,a,"config_"); print a[2]}')
export DTYPE="hmma"
export BENCHMARK="deepcam"
export FRAMEWORK="pytorch"

# API logging
export NCCL_TEST=0
export CHECK_COMPLIANCE=0
export APILOG_PRECISION="amp"
export APILOG_MODEL_NAME="deepcam"
