#!/bin/bash
#PBS -S /bin/bash
#PBS -M iyildirim@anl.gov
#PBS -A DLIO
#PBS -l filesystems=eagle:home
#PBS -l nodes=1:ppn=4
#PBS -l walltime=60:00
#PBS -q debug-scaling
#PBS -o logs/
#PBS -e logs/

set -x

cd $PBS_O_WORKDIR

module use /soft/modulefiles
module load conda/2024-04-29
module load cudatoolkit-standalone/11
module unload darshan

dataset_dir=${DATASET_DIR:-"/eagle/DLIO/iyildirim/datasets/criteo1tb_final2"}
work_dir=${WORKDIR:-"/home/iyildirim/projects/dlio_ml_workloads"}
dlrm_dir="$work_dir/dlrm"
training_dir="$dlrm_dir/mlcommons_training/recommendation_v2/torchrec_dlrm"

conda activate "$dlrm_dir/conda_env/"

# Environment settings
job_id=$(echo $PBS_JOBID | cut -d "." -f 1)
nhosts=$(cat $PBS_NODEFILE | uniq | sed -n $=)
ngpu_per_host=$(nvidia-smi -L | wc -l)
ngpus="$((${nhosts} * ${ngpu_per_host}))"
ppn=$ngpu_per_host

# Output settings
output_dir="$dlrm_dir/results/$job_id"
mkdir -p "$output_dir"

# Trace settings
export DFTRACER_DATA_DIR="$dataset_dir"
export DFTRACER_DISABLE_IO=0
export DFTRACER_DISABLE_POSIX=0
export DFTRACER_DISABLE_STDIO=0
export DFTRACER_ENABLE=${DFTRACER_ENABLE:-1}
export DFTRACER_INC_METADATA=${DFTRACER_INC_METADATA:-1}
export DFTRACER_LOG_FILE="$output_dir/trace"
# export DFTRACER_LOG_LEVEL=${DFTRACER_LOG_LEVEL:-"DEBUG"}

# Job settings
export MASTER_ADDR=$(cat $PBS_NODEFILE | head -n 1)
export MASTER_PORT=29500 # random port
export NCCL_COLLNET_ENABLE=1
# export NCCL_DEBUG=INFO                # debugging
# export TORCH_DISTRIBUTED_DEBUG=DETAIL # debugging

# Run settings
batch_size=${BATCH_SIZE:-8192}
embedding_dim=${EMBEDDING_DIM:-128}
epochs=${EPOCHS:-3}
learning_rate=${LEARNING_RATE:-0.1}
echo "{
    \"batch_size\": $batch_size,
    \"embedding_dim\": $embedding_dim,
    \"epochs\": $epochs,
    \"learning_rate\": $learning_rate,
    \"nhosts\": $nhosts,
    \"ngpus\": $ngpus,
    \"ngpu_per_host\": $ngpu_per_host,
    \"ppn\": $ppn
}" >"$output_dir/config.json"
nvidia-smi >$output_dir/gpu.txt
env >&$output_dir/env.txt

mpiexec -np $ngpus --ppn $ppn --cpu-bind depth -d $((64 / ppn)) "$dlrm_dir/launcher.sh" python3 "$training_dir/dlrm_main.py" \
    --batch_size $batch_size \
    --dense_arch_layer_sizes "512,256,128" \
    --embedding_dim $embedding_dim \
    --epochs $epochs \
    --evaluate_on_epoch_end \
    --evaluate_on_training_end \
    --in_memory_binary_criteo_path "$dataset_dir" \
    --learning_rate $learning_rate \
    --num_embeddings_per_feature "45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35" \
    --output_dir "$output_dir" \
    --over_arch_layer_sizes "1024,1024,512,256,1" \
    --pin_memory \
    --print_sharding_plan \
    --shuffle_batches

trace_name="dlrm-n$nhosts-ppn$ppn-gpu$ngpus-ep$epochs-bs$batch_size-ed$embedding_dim-lr$learning_rate"

python3 "$work_dir/pfw_utils/pfw2perfetto.py" -l "$output_dir" -o "$output_dir/$trace_name.pfw"
python3 "$work_dir/pfw_utils/pfw2perfetto.py" -l "$output_dir" -o "$output_dir/$trace_name-posix.pfw" --posix

cp "$output_dir/$trace_name.pfw" "$output_dir/../"
cp "$output_dir/$trace_name-posix.pfw" "$output_dir/../"
