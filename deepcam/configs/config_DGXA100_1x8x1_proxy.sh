#!/bin/bash

# data directory
source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_common.sh
export DATADIR=${DATADIR_SMALL}

# hyperparameters
export LOCAL_BATCH_SIZE=1
export START_LR=0.0055
export OPTIMIZER="DistributedLAMB"
export LR_SCHEDULE_TYPE="multistep"
export LR_MILESTONES="800"
export LR_DECAY_RATE="0.1"
export LR_WARMUP_STEPS=400
export LR_WARMUP_FACTOR=1.
export WEIGHT_DECAY=0.01
export BATCHNORM_GROUP_SIZE=2

# data parameters
export SHUFFLE_MODE="global"
export DATA_FORMAT="dali-es-gpu"
export DATA_OVERSAMPLING_FACTOR=2
export PRECISION_MODE="amp"
export LOCAL_VALIDATION_BATCH_SIZE=8
export MAX_THREADS=8

# misc args
export MAX_EPOCHS=10
export ADDITIONAL_ARGS="${ADDITIONAL_ARGS} --disable_comm_overlap --enable_graph --enable_groupbn --target_iou=1.01"
export CHECK_COMPLIANCE=0

# system parameters
export DGXNNODES=1
WALLTIME_MINUTES=15
export WALLTIME=$(( 15 + (${NEXP} * ${WALLTIME_MINUTES}) ))
#export SBATCH_NETWORK="sharp"
