source $(dirname ${BASH_SOURCE[0]})/config_DGXH100_common.sh

export STAGING_DIR="/raid/scratch"
export CONFIG_FILE="submission_dgxh100_1x8x1.yaml"

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME=40

export NCCL_TEST=0
export CHECK_COMPLIANCE=0

# dltools
export CONFIG="DGXH100_apilog_1x8x1"
export DTYPE="hmma"
export BENCHMARK="cosmoflow"
export FRAMEWORK="pytorch"
export APILOG_PRECISION="amp"
export APILOG_MODEL_NAME="cosmoflow"
