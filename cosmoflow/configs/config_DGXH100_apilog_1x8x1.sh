source $(dirname ${BASH_SOURCE[0]})/config_DGXH100_common.sh

export STAGING_DIR="/raid/scratch"
export CONFIG_FILE="submission_dgxh100_1x8x1.yaml"

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

WALLTIME_MINUTES=10
export WALLTIME=$(( 15 + (${NEXP:-1} * ${WALLTIME_MINUTES}) ))

