#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=4
#PBS -l walltime=1:00:00
#PBS -M huihuo.zheng@anl.gov
#PBS -A datascience
#PBS -l filesystems=grand:home
#PBS -q debug-scaling
cd $PBS_O_WORKDIR
module load conda
conda activate mlperf

PBS_JOBSIZE=0
for n in `get_hosts.py`
do
    PBS_JOBSIZE=$((PBS_JOBSIZE+1))
done
	 
export TAG=$(date +"%Y-%m-%d-%H-%M-%S")
NUM_WORKERS=4
PPN=4
OUTPUT=results/${TAG}/lustre
mkdir -p $OUTPUT

DATA_DIR=data NUM_WOKRERS=${NUM_WORKERS} OUTPUT_DIR=${OUTPUT} NPROC=$((PBS_JOBSIZE*PPN)) PPN=${PPN} ./run_polaris.sh
cp -rf data /local/scratch/
OUTPUT=results/${TAG}/nvme
mkdir -p $OUTPUT
DATA_DIR=/local/scratch/data NUM_WOKRERS=${NUM_WORKERS} OUTPUT_DIR=${OUTPUT} NPROC=$((PBS_JOBSIZE*PPN)) PPN=${PPN} ./run_polaris.sh 
