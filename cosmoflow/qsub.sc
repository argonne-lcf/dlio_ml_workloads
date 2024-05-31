#!/bin/bash
#PBS -S /bin/bash
#PBS -l walltime=1:00:00
#PBS -l nodes=2:ppn=4
#PBS -M huihuo.zheng@anl.gov
#PBS -q S1948341 -A GPU_Hack
#PBS -l filesystems=home:eagle
cd $PBS_O_WORKDIR
export NNODES=$(cat $PBS_NODEFILE | uniq | sed -n $=)
ID=$(echo $PBS_JOBID | cut -d "." -f 1)
source ./setup.sh
export TAG=$(date +"%Y-%m-%d-%H-%M-%S")

mpiexec -np $((NNODES*4)) --ppn 4 --cpu-bind depth -d 16 python ./main.py \
      +mpi.local_size=4 \
      +log.timestamp=ms_${NNODES}x4 \
      +log.experiment_id=${PBS_JOBID} \
      ++data.stage=/local/scratch/ \
      ++data.root_dir=/home/hzheng/datascience_grand/mlperf_hpc/hpc-nvidia/datasets/cosmoflow/tf_v2_gzip_256 \
      ++hydra.run.dir=results/${NNODES}x4/$TAG/ \
      ++model.training.train_epochs=2 \
      --config-name submission_dgxa100_2x8x1 
nvidia-smi > results/${NNODES}x4/$TAG/gpu.info
env >&  results/${NNODES}x4/$TAG/env.dat
cp $0.o$ID $0.e$ID ${OUTPUT}/
