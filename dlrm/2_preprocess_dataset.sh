#!/bin/bash
#PBS -S /bin/bash
#PBS -M iyildirim@anl.gov
#PBS -A DLIO
#PBS -l filesystems=eagle:home
#PBS -l nodes=1:ppn=4
#PBS -l walltime=60:00
#PBS -q debug
#PBS -o logs/
#PBS -e logs/

module use /soft/modulefiles
module load conda/2024-04-29
module load cudatoolkit-standalone/11
module unload darshan

dataset_dir=${DATASET_DIR:-"/eagle/DLIO/iyildirim/datasets"}
work_dir=${WORKDIR:-"/home/iyildirim/projects/dlio_ml_workloads"}
dlrm_dir="$work_dir/dlrm"

training_name="mlcommons_training"
training_dir="$work_dir/$training_name"
training_dlrm_dir="recommendation_v2/torchrec_dlrm"

conda activate "$dlrm_dir/conda_env/"

bash $training_dir/$training_dlrm_dir/scripts/process_Criteo_1TB_Click_Logs_dataset.sh \
    "$dataset_dir/criteo1tb_split" \
    "/local/scratch/criteo1tb_tmp" \
    "$dataset_dir/criteo1tb_final"
