#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=4
#PBS -l walltime=1:00:00
#PBS -M huihuo.zheng@anl.gov
#PBS -A datascience
#PBS -l filesystems=tegu:home
#PBS -q debug

cd $PBS_O_WORKDIR
source $HOME/dlio_ml_workloads/setup_ml_env.sh

export DATA_TOP=${WORKDIR}/unet3d/
[ -e ${DATA_TOP}/data ] || ln -s ${WORKDIR}/datasets/unet3d ${DATA_TOP}/data
export SKIP_REDUCE=1

PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | sed -n $=)
ID=$(echo $PBS_JOBID | cut -d "." -f 1)	 
export TAG=$(date +"%Y-%m-%d-%H-%M-%S").$ID

NUM_WORKERS=4
PPN=${PPN:-4}
export BATCH_SIZE=7
OUTPUT=results/n${PBS_JOBSIZE}.g${PPN}.b${BATCH_SIZE}/${TAG}/
mkdir -p $OUTPUT
export NPROC=$((PBS_JOBSIZE*PPN))
[ -e ${DATA_TOP}/datax$NPROC ] || aprun -n $NPROC -N $PPN --cc depth -d 16 ./launcher.sh ./mul.sh $NPROC
BATCH_SIZE=${BATCH_SIZE} DATA_DIR=${DATA_TOP}/datax$NPROC NUM_WORKERS=${NUM_WORKERS} OUTPUT_DIR=${OUTPUT} NPROC=$((PBS_JOBSIZE*PPN)) PPN=${PPN} ./run_polaris.sh | tee -a ${OUTPUT}/output.log
nvidia-smi > $OUTPUT/gpu.info
env >& $OUTPUT/env.dat
cp $0.o$ID $0.e$ID ${OUTPUT}/
