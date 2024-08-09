#!/bin/bash
#PBS -S /bin/bash
#PBS -M iyildirim@anl.gov
#PBS -A DLIO
#PBS -l filesystems=eagle:home
#PBS -l nodes=1:ppn=1
#PBS -l walltime=30:00
#PBS -q debug
#PBS -o logs/
#PBS -e logs/

module use /soft/modulefiles
module load conda/2024-04-29
module load cudatoolkit-standalone/11
module unload darshan

work_dir="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
torchrec_name="torchrec"
torchrec_dir="$work_dir/$torchrec_name"
torchrec_version="0.3.2"
training_name="mlcommons_training"
training_dir="$work_dir/$training_name"
training_dlrm_dir="recommendation_v2/torchrec_dlrm"
venv_dir="$work_dir/conda_env/"

if [ ! -d "$torchrec_dir" ]; then
    git clone --branch v$torchrec_version https://github.com/pytorch/torchrec.git $torchrec_dir
    cd $torchrec_dir/
    git apply "$work_dir/$torchrec_name.patch"
    cd -
fi

if [ ! -d "$training_dir" ]; then
    git clone --no-checkout https://github.com/mlcommons/training.git $training_dir
    cd $training_dir/
    git sparse-checkout init
    git sparse-checkout set $training_dlrm_dir
    git checkout
    git apply "$work_dir/$training_name.patch"
    cd -
fi

if [ ! -d "$venv_dir" ]; then
    conda activate base
    conda create -y --prefix $venv_dir python=3.7
    conda activate $venv_dir
    python -m pip install -r $training_dir/$training_dlrm_dir/requirements.txt
    python -m pip install -e $torchrec_dir
    python -m pip install pydftracer
else
    conda activate $venv_dir
fi

# img="dlrm.sif"
# img_temp="dlrm_temp.sif"

# if [ ! -d "$img_temp" ]; then
#     apptainer build --fakeroot --sandbox $img_temp dlrm.def
# fi

# if [ ! -e "$img" ]; then
#     apptainer exec --writable -B $workspace_dir:/workspace $img_temp \
#     python -m pip install -r /workspace/$training_dir/$training_dlrm_dir/requirements.txt
#     apptainer exec --writable -B $workspace_dir:/workspace $img_temp \
#     python -m pip install -e /workspace/$torchrec_dir
#     apptainer build --fakeroot $img $img_temp
# fi

# apptainer exec --nv $img nvidia-smi

# apptainer exec -B $workspace_dir:/workspace -B $dataset_dir

# apptainer exec --nv -B .:/workspace -B /eagle/DLIO/iyildirim dlrm.sif python /workspace/preproc/torchrec/datasets/scripts/npy_preproc_criteo.py --input_dir /eagle/DLIO/iyildirim/datasets/criteo1tb_split/ --output_dir /eagle/DLIO/iyildirim/scratch/criteo1tb/
