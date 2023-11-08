#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=4
#PBS -l walltime=1:00:00
#PBS -M huihuo.zheng@anl.gov
#PBS -A datascience
#PBS -l filesystems=grand:home
#PBS -q debug

cd $PBS_O_WORKDIR
source $HOME/PolarisAT/dlio_ml_workloads/setup_ml_env.sh
export DATA_TOP=/home/hzheng/PolarisAT/dlio_ml_workloads/unet3d
export SKIP_REDUCE=1

PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | sed -n $=)
	 
export TAG=$(date +"%Y-%m-%d-%H-%M-%S").$PBS_JOBID

NUM_WORKERS=4
PPN=1
export BATCH_SIZE=4
OUTPUT=results/n1.g${PPN}.b${BATCH_SIZE}/${TAG}/
mkdir -p $OUTPUT
BATCH_SIZE=${BATCH_SIZE} DATA_DIR=${DATA_TOP}/datax$((PBS_JOBSIZE*PPN)) NUM_WOKRERS=${NUM_WORKERS} OUTPUT_DIR=${OUTPUT} NPROC=$((PBS_JOBSIZE*PPN)) PPN=${PPN} ./run_polaris.sh


PPN=4
OUTPUT=results/n1.g${PPN}.b${BATCH_SIZE}/${TAG}/
mkdir -p $OUTPUT
BATCH_SIZE=${BATCH_SIZE} DATA_DIR=${DATA_TOP}/datax$((PBS_JOBSIZE*PPN)) NUM_WOKRERS=${NUM_WORKERS} OUTPUT_DIR=${OUTPUT} NPROC=$((PBS_JOBSIZE*PPN)) PPN=${PPN} ./run_polaris.sh
