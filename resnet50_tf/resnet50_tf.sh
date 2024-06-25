#!/bin/bash
#PBS -l nodes=1:ppn=4
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A datascience
#PBS -N resnet50_tf
#PBS -l filesystems=home:eagle

cd $PBS_O_WORKDIR
export WORKDIR=$HOME/dlio_ml_workloads
source ${WORKDIR}/setup_ml_env.sh
export PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | wc -l)
ID=$(echo $PBS_JOBID | cut -d "." -f 1)	 
export TAG=$(date +"%Y-%m-%d-%H-%M-%S").$ID
export PPN=${PPN:-4}
export BATCH_SIZE=400
export OUTPUT=results/n${PBS_JOBSIZE}.g${PPN}.b${BATCH_SIZE}/$TAG
mkdir -p $OUTPUT
echo "Started at `date`"
mpiexec -np $((PBS_JOBSIZE*PPN)) --ppn ${PPN} --cpu-bind depth -d 16 python resnet50_hvd.py \
	--datagen tfrecord \
	--data-folder /eagle/datasets/ImageNet/tfrecords/ \
	--batch-size ${BATCH_SIZE} --num-workers 8 --epochs 5 --steps 100 --output-folder $OUTPUT | tee -a $OUTPUT/output.log
echo "Ended at `date`"
