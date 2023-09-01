#!/bin/bash
#PBS -S /bin/bash
#PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=4
#PBS -M huihuo.zheng@anl.gov
#PBS -A datascience
#PBS -q debug
#PBS -l filesystems=home:grand:eagle
cd /home/hzheng/PolarisAT/dlio_ml_workloads/cosmoflow
function getrank()
{
    return ${PMI_LOCAL_RANK}
}
source ./setup.sh
export PBS_JOBSIZE=$(cat $PBS_NODEFILE | sort | uniq | sed -n $=)
export PPN=${PPN:-4}
export TAG=$(date +"%Y-%m-%d-%H-%M-%S")
export DONT_USE_MMAP=${DONT_USE_MMAP:-True}
export CONFIG=${CONFIG:-"submission_dgxa100_2x8x1"}
aprun --cc depth -e OMP_NUM_THREADS=$((64/PPN)) -d $((64/PPN)) -n $((PBS_JOBSIZE*PPN)) -N ${PPN} \
      python ./main.py +mpi.local_size=${PPN} ++data.stage=/local/scratch/ \
      ++data.dont_use_mmap=${DONT_USE_MMAP} \
      +log.timestamp=ms_${CONFIG} \
      ++model.training.train_epochs=1 \
      +model.training.train_steps=1000 \
      +log.experiment_id=${PBS_JOBID} \
      ++data.root_dir="/home/hzheng/datascience_grand/mlperf_hpc/hpc-nvidia/datasets/cosmoflow/tf_v2_256/" \
      ++hydra.run.dir=results/${TAG}/${PBS_JOBSIZE}x${PPN} \
      --config-name ${CONFIG}
