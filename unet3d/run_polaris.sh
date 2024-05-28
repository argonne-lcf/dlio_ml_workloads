#!/bin/bash +x
#COBALT -q gpu_a100 -t 1:00:00 -n 1
# The following seven lines are specific to Argonne JLSE. Please comment them 
#module load darshan/darshan-openmpi-gcc
source /home/hzheng/PolarisAT_eagle/dlio_ml_workloads/setup_ml_env.sh
export RDMAV_HUGEPAGES_SAFE=1
set -e
free -h 
# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>

SEED=${1:--1}

#MAX_EPOCHS=4000
MAX_EPOCHS=${EPOCHS:-5}
QUALITY_THRESHOLD="0.908"
START_EVAL_AT=10
EVALUATE_EVERY=8
LEARNING_RATE="0.8"
LR_WARMUP_EPOCHS=200
DATASET_DIR=${DATA_DIR:-"/eagle/datasets/unet3d/"}
BATCH_SIZE=${BATCH_SIZE:-7}
GRADIENT_ACCUMULATION_STEPS=1
NUM_WORKERS=${NUM_WORKERS:-4}
SLEEP=${SLEEP:--1}
OUTPUT_DIR=${OUTPUT_DIR:-"results/"}
NPROC=${NPROC:-1}
PPN=${PPN:-4}
echo "{
 MAX_EPOCHS: ${MAX_EPOCHS}, 
 QUALITY_THRESHOLD: ${QUALITY_THRESHOLD},
 START_EVAL_AT: ${START_EVAL_AT},
 EVALUATE_EVERY: ${EVALUATE_EVERY},
 LEARNING_RATE: ${LEARNING_RATE},
 LR_WARMUP_EPOCHS: ${LR_WARMUP_EPOCHS},
 DATASET_DIR: ${DATASET_DIR},
 BATCH_SIZE: ${BATCH_SIZE}, 
 GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS},
 NUM_WORKERS: ${NUM_WORKERS},
 HOSTNAME: ${HOSTNAME},
 SLEEP: ${SLEEP},
 OUTPUT_DIR: ${OUTPUT_DIR},
}" >& ${OUTPUT_DIR}/config.json

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
#export LD_PRELOAD=$HOME//PolarisAT/pyenvs/ml_workload/lib/libdlio_profiler_preload.so
aprun --cc depth -n ${NPROC} -N ${PPN} -d $((64/PPN)) -e OMP_NUM_THREADS=$((64/PPN)) ./launcher.sh python3 main.py --data_dir ${DATASET_DIR} \
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
    --num_workers ${NUM_WORKERS} \
    --output_dir ${OUTPUT_DIR} \
    --sleep ${SLEEP} 
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
