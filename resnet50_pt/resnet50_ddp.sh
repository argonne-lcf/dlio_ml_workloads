#!/bin/bash
#PBS -l nodes=2:ncpus=4
#PBS -l walltime=1:00:00
#PBS -A datascience
#PBS -l filesystems=home:tegu

cd $PBS_O_WORKDIR
export WORKDIR=$HOME/dlio_ml_workloads
source ${WORKDIR}/setup_ml_env.sh

export TORCH_PROFILER_ENABLE=1

ID=$(echo $PBS_JOBID | cut -d "." -f 1)	 
export TAG=$(date +"%Y-%m-%d-%H-%M-%S").$ID
export PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | wc -l)
export PPN=${PPN:-4}
export BATCH_SIZE=400
export OUTPUT=results/n${PBS_JOBSIZE}.g${PPN}.b${BATCH_SIZE}/$TAG
mkdir -p $OUTPUT

mpiexec -np $((PBS_JOBSIZE*PPN)) --ppn ${PPN} --cpu-bind depth -d 16 python resnet50_ddp.py --batch-size ${BATCH_SIZE} --output-folder $OUTPUT  --dummy | tee -a $OUTPUT/output.log
