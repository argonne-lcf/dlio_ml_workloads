#!/bin/bash -l
#PBS -l walltime=00:30:00
#PBS -l nodes=4:ppn=4
#PBS -q debug-scaling
#PBS -A datascience
#PBS -N deepcam
#PBS -l filesystems=home:grand

cd $PBS_O_WORKDIR

#source /home/hzheng/PolarisAT/dlio_ml_workloads/setup_ml_env.sh
DATE_TAG=2023-10-04 source /home/hzheng/PolarisAT/dlio_ml_workloads/setup_ml_env.sh  

#module load singularity

NODES=$(cat $PBS_NODEFILE | uniq | wc -l)
GPUS_PER_NODE=4
RANKS=$((NODES * GPUS_PER_NODE))
echo NODES=$NODES  PPN=$GPUS_PER_NODE  RANKS=$RANKS
export PYTHONPATH=/home/hzheng/PolarisAT/dlio_ml_workloads:$PYTHONPATH
export APPDIR=$PWD
export TAG=$(date +"%Y-%m-%d-%H-%M-%S")
export OUTPUT_DIR=$PWD/results/${NODES}x${GPUS_PER_NODE}/$TAG/

# Enable GPU-MPI (if supported by application)
export MPICH_GPU_SUPPORT_ENABLED=1

# MPI and OpenMP settings
NNODES=$(cat $PBS_NODEFILE | uniq | wc -l)
NRANKS_PER_NODE=4
NDEPTH=8
NTHREADS=1
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))

echo "which python:" $(which python)
echo "python path:" $PYTHONPATH


### loop for 5 iterations
echo "Start of iteration number : $iter"

###### other parameters
export SEED=${_seed_override:-$(date +%s)}
export DATESTAMP=$(date +'%y%m%d%H%M%S%N')
export RUN_TAG=${RUN_TAG:-${DATESTAMP}}
export WIREUP_METHOD="nccl-openmpi"
unset TRAINING_INSTANCE_SIZE

###### source config file
source $APPDIR/configs/config_polaris_128x4x2.sh
export DGXNGPU=${GPUS_PER_NODE}

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# assemble launch command
export TOTALGPUS=$RANKS
if [ ! -z ${TRAINING_INSTANCE_SIZE} ]; then
    gpu_config="$(( ${TOTALGPUS} / ${TRAINING_INSTANCE_SIZE} ))x${TRAINING_INSTANCE_SIZE}"
else
    gpu_config=${TOTALGPUS}
fi

# create tmp directory
mkdir -p ${OUTPUT_DIR}

# LR switch
if [ -z ${LR_SCHEDULE_TYPE} ]; then
    lr_schedule_arg=""
elif [ "${LR_SCHEDULE_TYPE}" == "multistep" ]; then
    lr_schedule_arg="--lr_schedule type=${LR_SCHEDULE_TYPE},milestones=${LR_MILESTONES},decay_rate=${LR_DECAY_RATE}"
elif [ "${LR_SCHEDULE_TYPE}" == "cosine_annealing" ]; then
    lr_schedule_arg="--lr_schedule type=${LR_SCHEDULE_TYPE},t_max=${LR_T_MAX},eta_min=${LR_ETA_MIN}"
fi

PARAMS=(
    --wireup_method ${WIREUP_METHOD}
    --run_tag ${RUN_TAG}
    --experiment_id ${EXP_ID:-1}
    --data_dir_prefix ${DATA_DIR_PREFIX:-"/lus/grand/projects/datascience/MlPerf-datasets/deepcam/All-Hist-numpy/"}
    --output_dir ${OUTPUT_DIR}
    --model_prefix "segmentation"
    --optimizer ${OPTIMIZER}
    --start_lr ${START_LR}
    ${lr_schedule_arg}
    --lr_warmup_steps ${LR_WARMUP_STEPS}
    --lr_warmup_factor ${LR_WARMUP_FACTOR}
    --weight_decay ${WEIGHT_DECAY}
    --logging_frequency ${LOGGING_FREQUENCY}
    --save_frequency 100000
    --max_epochs ${MAX_EPOCHS:-200}
    --seed ${SEED}
    --batchnorm_group_size ${BATCHNORM_GROUP_SIZE}
    --shuffle_mode "${SHUFFLE_MODE}"
    --data_format "${DATA_FORMAT}"
    --data_oversampling_factor ${DATA_OVERSAMPLING_FACTOR:-1}
    --precision_mode "${PRECISION_MODE}"
    --enable_nhwc
    --local_batch_size ${LOCAL_BATCH_SIZE}
    --local_batch_size_validation ${LOCAL_VALIDATION_BATCH_SIZE}
    --data_num_threads ${MAX_THREADS:-4}
    ${ADDITIONAL_ARGS}
)

# change directory
#pushd /opt/deepCam
cd $APPDIR

# profile command:
if [ ! -z ${OMPI_COMM_WORLD_RANK} ]; then
    WORLD_RANK=${OMPI_COMM_WORLD_RANK}
elif [ ! -z ${PMIX_RANK} ]; then
    WORLD_RANK=${PMIX_RANK}
elif [ ! -z ${PMI_RANK} ]; then
    WORLD_RANK=${PMI_RANK}
fi
PROFILE_BASE_CMD="nsys profile --mpi-impl=openmpi --trace=osrt,cuda,cublas,nvtx,mpi --kill none -c cudaProfilerApi -f true -o ${OUTPUT_DIR}/profile_job${SLURM_JOBID}_rank${WORLD_RANK}"
ANNA_BASE_CMD="nsys profile --trace cuda,nvtx --sample cpu --output ${OUTPUT_DIR}/anna_job${SLURM_JOBID}_rank${WORLD_RANK} --export sqlite --force-overwrite true --stop-on-exit true --capture-range cudaProfilerApi --capture-range-end stop --kill none"
DLPROF_BASE_CMD="dlprof --mode=pytorch --force=true --reports=summary,detail,iteration --nsys_profile_range=true --output_path=${OUTPUT_DIR} --profile_name=dlprof_rank${WORLD_RANK}"
METRICS_BASE_CMD="ncu --target-processes=all --profile-from-start=off --nvtx --print-summary=per-nvtx --csv -f -o ${OUTPUT_DIR}/metrics_rank${WORLD_RANK} --metrics=smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__inst_executed_pipe_tensor.sum"

if [[ ${ENABLE_PROFILING} == 1 ]]; then
    if [[ ${ENABLE_METRICS_COLLECTION} == 1 ]]; then
	echo "Metric Collection enabled"
	if [[ "${WORLD_RANK}" == "0" ]]; then
	    PROFILE_CMD=${METRICS_BASE_CMD}
	else
	    PROFILE_CMD=""
	fi
    elif [[ ${ENABLE_DLPROF} == 1 ]]; then
	echo "Dlprof enabled"
	if [[ "${WORLD_RANK}" == "0" ]]; then
	    PROFILE_CMD=${DLPROF_BASE_CMD}
	else
	    PROFILE_CMD=""
	fi
	PARAMS+=(--profile_markers=dlprof)
    elif [[ ${ENABLE_ANNA} == 1 ]]; then
	echo "ANNA enabled"
	if [[ "${WORLD_RANK}" == "0" ]]; then
	    PROFILE_CMD=${ANNA_BASE_CMD}
	else
	    PROFILE_CMD=""
	fi
	PARAMS+=(--profile_markers=anna)
    else
	echo "Profiling enabled"
	PROFILE_CMD=${PROFILE_BASE_CMD}
    fi
else
    PROFILE_CMD=""
fi

if [[ ${DEBUG_MEMCHECK} == 1 ]]; then
    echo "Debugging enabled"
    DEBUG_CMD="compute-sanitizer --tool=memcheck"
else
    DEBUG_CMD=""
fi

# bind command

# BIND_CMD="./src/deepCam/bind.sh --cluster=selene --ib=single --cpu=exclusive"
BIND_CMD=""

# do we cache data
if [ ! -z ${DATA_CACHE_DIRECTORY} ]; then
    PARAMS+=(--data_cache_directory ${DATA_CACHE_DIRECTORY})
fi

# run script selection:
if [ ! -z ${TRAINING_INSTANCE_SIZE} ]; then
    echo "Running Multi Instance Training"
    RUN_SCRIPT="$APPDIR/train_instance_oo.py"
    PARAMS+=(--training_instance_size ${TRAINING_INSTANCE_SIZE})

    if [ ! -z ${STAGE_DIR_PREFIX} ]; then
	PARAMS+=(
	    --stage_dir_prefix ${STAGE_DIR_PREFIX}
	    --stage_num_read_workers ${STAGE_NUM_READ_WORKERS:-1}
	    --stage_num_write_workers ${STAGE_NUM_WRITE_WORKERS:-1}
	    --stage_batch_size ${STAGE_BATCH_SIZE:--1}
	    --stage_mode ${STAGE_MODE:-"node"}
	)
	# do we need to verify the staging results
	if [ "${STAGE_VERIFY:-0}" -eq 1 ]; then
	    PARAMS+=(--stage_verify)
	fi
	if [ "${STAGE_ONLY:-0}" -eq 1 ]; then
	    echo "WARNING: You are about to run a staging only benchmark"
	    PARAMS+=(--stage_only)
	fi
	if [ "${STAGE_FULL_DATA_PER_NODE:-0}" -eq 1 ]; then
	    PARAMS+=(--stage_full_data_per_node)
	fi
	if [ "${STAGE_ARCHIVES:-0}" -eq 1 ]; then
	    PARAMS+=(--stage_archives)
	fi
	if [ "${STAGE_USE_DIRECT_IO:-0}" -eq 1 ]; then
	    PARAMS+=(--stage_use_direct_io)
	fi
	if [ "${STAGE_READ_ONLY:-0}" -eq 1 ]; then
	    PARAMS+=(--stage_read_only)
	fi
    fi
else
    echo "Running Single Instance Training"
    RUN_SCRIPT="$APPDIR/src/deepCam/train.py"
fi

# decide whether to enable profiling
if [ ! -z ${ENABLE_PROFILING} ] && [ ${ENABLE_PROFILING} == 1 ]; then
    echo "Running Profiling"
    if [ ! -z ${TRAINING_INSTANCE_SIZE} ]; then
	RUN_SCRIPT="$APPDIR/train_instance_oo_profile.py"
    else
	RUN_SCRIPT="$APPDIR/train_profile.py"
    fi

    if [ ! -z ${CAPTURE_RANGE_START} ]; then
	PARAMS+=(
	    --capture_range_start ${CAPTURE_RANGE_START}
	    --capture_range_stop ${CAPTURE_RANGE_STOP}
	)
    fi

    if [ ! -z ${PROFILE_FRACTION} ]; then
	PARAMS+=(--profile_fraction ${PROFILE_FRACTION})
    fi
fi

# assemble run command
RUN_CMD="${RUN_SCRIPT} ${PARAMS[@]}"

# run command
# ${BIND_CMD} ${PROFILE_CMD} ${DEBUG_CMD} 

aprun -n ${NTOTRANKS} -N ${NRANKS_PER_NODE} --cc depth -d 16 $(which python) ${RUN_CMD}; ret_code=$? 
#if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# cleanup command
#CLEANUP_CMD="cp -r ${OUTPUT_DIR}/* /results/"
#${CLEANUP_CMD} 

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
# report result
result=$(( $end - $start ))
result_name="DEEPCAM_HPC"
echo "RESULT,$result_name,,$result,$USER,$start_fmt"

echo "End of $iter run"
echo "---------------"
