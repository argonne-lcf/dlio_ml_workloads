#!/bin/bash                                                                                                                                                         

# data directory
source $(dirname ${BASH_SOURCE[0]})/config_DGXH100_common.sh

# hyperparameters
export LOCAL_BATCH_SIZE=1
export START_LR=0.0062
export OPTIMIZER="MixedPrecisionLAMB"
export LR_SCHEDULE_TYPE="cosine_annealing"
export LR_T_MAX="2600"
export LR_ETA_MIN="0.0"
export LR_WARMUP_STEPS=400
export LR_WARMUP_FACTOR=1.
export WEIGHT_DECAY=0.01
export BATCHNORM_GROUP_SIZE=2

# data parameters
export SHUFFLE_MODE="global"
export DATA_FORMAT="dali-es-disk"
export PRECISION_MODE="amp"
export LOCAL_VALIDATION_BATCH_SIZE=8

# misc args
export ADDITIONAL_ARGS="--enable_graph --disable_comm_overlap --enable_mmap --enable_groupbn --synchronous_staging"

# in case of power management, run for more epochs
export MAX_EPOCHS=90

# number of experiments
export NEXP=1

# optimal frequencies
export MAXQ_CLK=1200
export MINEDP_CLK=1245

# system parameters
export DGXNGPU=8
export DGXNNODES=64

if [[ ${MLPERF_POWER_TRAIN_AFTER_RUN_STOP:-0} -eq 1 ]]; then
    WALLTIME_MINUTES=20
else
    WALLTIME_MINUTES=5
fi
export WALLTIME=$(( 15 + (${NEXP} * ${WALLTIME_MINUTES}) ))
