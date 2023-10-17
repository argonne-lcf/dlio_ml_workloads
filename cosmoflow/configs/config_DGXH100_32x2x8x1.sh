source $(dirname ${BASH_SOURCE[0]})/config_DGXH100_common.sh

export STAGING_DIR="/raid/scratch"
export CONFIG_FILE="submission_dgxh100_2x8x1.yaml"
export NUM_INSTANCES=32

## System run parms
export DGXNNODES=64
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

WALLTIME_MINUTES=60
export WALLTIME=$(( 15 + (${NEXP:-1} * ${WALLTIME_MINUTES}) ))

