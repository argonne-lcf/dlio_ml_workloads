#!/bin/bash

export DATADIR="/lustre/fsw/mlperft/data/cosmoflow/cosmoflow_gzip"

# JET
export JET_DIR="/project/mlperft/common/jet2"
export JET_UPLOAD="jet logs upload output.zip"
export JET_CREATE="jet logs create output.zip --fill-gpu --fill-cpu --fill-system --fill-libraries --data user=${USER} --data workload.maintainers=${USER} --data type=workload --data workload.type=custom --data origin=${JET_ORIGIN:-"mlperf-manual"} --data workload.spec.script=run_and_time.sh "
