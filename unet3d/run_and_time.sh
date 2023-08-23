#!/bin/bash
#COBALT -q gpu_a100 -t 1:00:00 -n 1
# The following seven lines are specific to Argonne JLSE. Please comment them 
 . /etc/profile.d/modules.sh
module use $HOME/soft/modulefiles
module load anaconda3
export LD_PRELOAD=$HOME/soft/darshan/lib/libdarshan.so
export DARSHAN_DISABLE_SHARED_REDUCTION=1
export DXT_ENABLE_IO_TRACE=4
export DARSHAN_LOG_DIR=./darshan_log

set -e
free -h 
# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>

SEED=${1:--1}

#MAX_EPOCHS=4000
MAX_EPOCHS=8
QUALITY_THRESHOLD="0.908"
START_EVAL_AT=1000
EVALUATE_EVERY=20
LEARNING_RATE="0.8"
LR_WARMUP_EPOCHS=200
DATASET_DIR="./data"
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1
NUM_WORKERS=1

if [ -d ${DATASET_DIR} ]
then
    # start timing
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"
    echo "Number of data loader workers: ${NUM_WORKERS}"
# CLEAR YOUR CACHE HERE
  python -c "
from mlperf_logging.mllog import constants
from runtime.logging import mllog_event
mllog_event(key=constants.CACHE_CLEAR, value=True)"

  python main.py --data_dir ${DATASET_DIR} \
    --epochs ${MAX_EPOCHS} \
    --evaluate_every ${EVALUATE_EVERY} \
    --start_eval_at ${START_EVAL_AT} \
    --quality_threshold ${QUALITY_THRESHOLD} \
    --batch_size ${BATCH_SIZE} \
    --optimizer sgd \
    --ga_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --seed ${SEED} \
    --lr_warmup_epochs ${LR_WARMUP_EPOCHS} \
    --num_workers ${NUM_WORKERS}
	# end timing
	end=$(date +%s)
	end_fmt=$(date +%Y-%m-%d\ %r)
	echo "ENDING TIMING RUN AT $end_fmt"


	# report result
	result=$(( $end - $start ))
	result_name="image_segmentation"


	echo "RESULT,$result_name,$SEED,$result,$USER,$start_fmt"
else
	echo "Directory ${DATASET_DIR} does not exist"
fi
