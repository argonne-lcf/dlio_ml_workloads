#!/bin/bash
#PBS -S /bin/bash
#PBS -l walltime=1:00:00
#PBS -l nodes=2:ppn=4
#PBS -M huihuo.zheng@anl.gov
#PBS -A datascience
#PBS -l filesystems=home:tegu
#PBS -N cosmoflow
cd $PBS_O_WORKDIR
export WORKDIR=$HOME/dlio_ml_workloads/
source $WORKDIR/setup_ml_env.sh
export TORCH_PROFILER_ENABLE=${TORCH_PROFILER_ENABLE:-0}
export DLIO_PROFILER_DISABLE_IO=1

export NNODES=$(cat $PBS_NODEFILE | uniq | sed -n $=)
ID=$(echo $PBS_JOBID | cut -d "." -f 1)
export TAG=$(date +"%Y-%m-%d-%H-%M-%S").$ID
export BATCH_SIZE=1
export PPN=${PPN:-4}
export OUTPUT=results/n${NNODES}.g$PPN.b${BATCH_SIZE}/$TAG/
mkdir -p $OUTPUT
mpiexec -np $((NNODES*PPN)) --ppn $PPN --cpu-bind depth -d 16 python ./main.py \
      +mpi.local_size=$PPN \
      +log.timestamp=ms_${NNODES}x$PPN \
      +log.experiment_id=${PBS_JOBID} \
      ++data.stage=/local/scratch/ \
      ++data.root_dir=${WORKDIR}/datasets/cosmoflow/tf_v2_gzip_256 \
      ++hydra.run.dir=${OUTPUT} \
      ++model.training.train_epochs=1 \
      --config-name submission_dgxa100_2x8x1 | tee -a ${OUTPUT}/output.log
nvidia-smi > ${OUTPUT}/gpu.info
env >&  ${OUTPUT}/env.dat
cp $0.o$ID $0.e$ID ${OUTPUT}/
