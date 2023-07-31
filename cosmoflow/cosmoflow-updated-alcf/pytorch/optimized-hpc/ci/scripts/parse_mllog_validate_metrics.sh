# TODO: take logdir that contains the mllog(filename is $DATESTAMP.log) and ../.env file and return fail/pass
LOGDIR=$1
export SAVE_TO_KIBANA=1
export ES_CONNECTION=0

export CODEDIR=$(pwd)

###################################################################################
# BENCHMARK After Script ##########################################################
###################################################################################

 set -x
  # environment of the training run is backed in .env
  source ${LOGDIR}/../.env
  # override variables for processing in setup diff than training
  # export LOGDIR=/home/senthils/summa/logdir
  # mkdir -p "${LOGDIR}"
  # export DATESTAMP="2024"
  # BENCHMARK single_stage_dectector is used for pytorch for mxnet following is the benchmark name used in baseline
  # export BENCHMARK="ssd_mxnet"

# variabled needed for mlperf->dllog->joclog->nvdataflow
 export TESTNAME="${DGXSYSTEM}"
 export TESTTYPE="perf-train"
 export LOGNAME="${LOGDIR}/${DATESTAMP}.log"
 export LOGENV="${LOGDIR}/${TESTNAME}_env.json"
 export LOGDLL="${LOGDIR}/${TESTNAME}.json"
 export METADATA_PATH="${CODEDIR}/ci/baselines/${BENCHMARK}-${FRAMEWORK}/metric_metadata.json"
 export BASELINE_PATH="${CODEDIR}/ci/baselines/${BENCHMARK}-${FRAMEWORK}/baselines.json"

 python3 ${CODEDIR}/ci/scripts/mlperf_to_dllog.py --file "${LOGNAME}" --logdir "${LOGDIR}" --env_json_path "${LOGENV}" --dll_path "${LOGDLL}" --benchmark "${BENCHMARK}" --system_config "${DGXSYSTEM}"
 cd ${CODEDIR}
 if [ "${SLURM_EXITCODE}" == "0" ] && [ "${SLURM_STATE}" == "COMPLETED" ]; then
     python3 -m ci.scripts.joc-ci-tools.log_ops.baseline \
         "${TESTNAME}" \
         "${TESTTYPE}" \
         "${CI_COMMIT_REF_NAME}" \
         "${LOGDLL}" \
         "${METADATA_PATH}" \
         "${BASELINE_PATH}" \
         --elastic_search_connection ${ES_CONNECTION} \
         --output_path "${LOGDIR}/${TESTNAME}_baseline.json"
 fi
