#!/bin/bash
#PBS -S /bin/bash
#PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=4
#PBS -M huihuo.zheng@anl.gov
#PBS -A datascience
#PBS -q debug
#PBS -l filesystems=home:grand
cd $PBS_O_WORKDIR
export NNODES=$(cat $PBS_NODEFILE | uniq | sed -n $=)

source ./setup.sh
source $HOME/datascience_grand/http_proxy_polaris
export TAG=$(date +"%Y-%m-%d-%H-%M-%S")
aprun -n $((NNODES*1)) -N 1 --cc depth -d 16 python ./main.py \
      +mpi.local_size=1 \
      +log.timestamp=ms_${NNODES}x1 \
      +log.experiment_id=${PBS_JOBID} \
      ++data.stage=False \
      ++data.dataset="cosmoflow_npy" \
      ++data.root_dir=/home/hzheng/datascience_grand/mlperf_hpc/hpc-nvidia/datasets/cosmoflow/npy_v2_256 \
      ++hydra.run.dir=results_npy/${NNODES}x1/$TAG/ \
      ++model.training.train_epochs=2 \
      --config-name submission_dgxa100_2x8x1 

aprun -n $((NNODES*4)) -N 4 --cc depth -d 16 python ./main.py \
      +mpi.local_size=4 \
      +log.timestamp=ms_${NNODES}x4 \
      +log.experiment_id=${PBS_JOBID} \
      ++data.stage=False \
      ++data.dataset="cosmoflow_npy" \
      ++data.root_dir=/home/hzheng/datascience_grand/mlperf_hpc/hpc-nvidia/datasets/cosmoflow/npy_v2_256 \
      ++hydra.run.dir=results_npy/${NNODES}x4/$TAG/ \
      ++model.training.train_epochs=2 \
      --config-name submission_dgxa100_2x8x1 
