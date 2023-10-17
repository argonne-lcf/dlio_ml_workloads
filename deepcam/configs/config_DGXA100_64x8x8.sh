#!/bin/bash                                                                                                                                                                  

# data directory
source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_common.sh

# hyperparameters
export LOCAL_BATCH_SIZE=8
export START_LR=0.005870758168390416
export OPTIMIZER="MixedPrecisionLAMB"
export LR_SCHEDULE_TYPE="multistep"
export LR_MILESTONES="600"
export LR_DECAY_RATE="0.1"
export LR_WARMUP_STEPS=400
export LR_WARMUP_FACTOR=1.
export WEIGHT_DECAY=0.02381391688111777
export BATCHNORM_GROUP_SIZE=1

# data parameters
export SHUFFLE_MODE="global"
export DATA_FORMAT="dali-es"
export DATA_OVERSAMPLING_FACTOR=1
export PRECISION_MODE="amp"
export LOCAL_VALIDATION_BATCH_SIZE=8

# misc args
export ADDITIONAL_ARGS="${ADDITIONAL_ARGS} --disable_comm_overlap --enable_graph"

# system parameters
export DGXNNODES=64
WALLTIME_MINUTES=15
export WALLTIME=$(( 15 + (${NEXP} * ${WALLTIME_MINUTES}) ))
