export CI_PIPELINE_ID=1801168
export CI_PROJECT_ID=9530
export SAVE_TO_KIBANA=1
export ES_CONNECTION=0

#export BENCHMARK="image_classification"
#export DATESTAMP="2020"
#export DGXSYSTEM="DGXA100"
#export FRAMEWORK="mxnet"
#export FW_VARIANCE=""
#export CI_JOB_ID="16872520"

# export BENCHMARK="language_model"
# export DATESTAMP="2021"
# export DGXSYSTEM="DGXA100_1x8x32x1"
# export FRAMEWORK="pytorch"
# export FW_VARIANCE=""
# export CI_JOB_ID="16872540"

#export BENCHMARK="reinforcement"
#export DATESTAMP="2022"
#export DGXSYSTEM="DGXA100"
#export FRAMEWORK="tensorflow"
#export FW_VARIANCE=""
#export CI_JOB_ID="16872546"

# export BENCHMARK="object_detection"
# export DATESTAMP="2023"
# export DGXSYSTEM="DGXA100"
# export FRAMEWORK="pytorch"
# export FW_VARIANCE=""
# export CI_JOB_ID="17153372"

export BENCHMARK="ssd_mxnet"
export DATESTAMP="2024"
export DGXSYSTEM="DGXA100_1x8x2x1--manual"
export FRAMEWORK="mxnet"
export FW_VARIANCE=""
export CI_JOB_ID="17153377"

#export BENCHMARK="translation"
#export DATESTAMP="2025"
#export DGXSYSTEM="DGXA100"
#export FRAMEWORK="pytorch"
#export FW_VARIANCE=""
#export CI_JOB_ID="16872605"

#export BENCHMARK="single_stage_detector"
#export DATESTAMP="2026"
#export DGXSYSTEM="DGXA100"
#export FRAMEWORK="pytorch"
#export FW_VARIANCE=""
#export CI_JOB_ID="16872592"

export LOGDIR=~/logdir
export GPFSFOLDER=~/GPFS
export CODEDIR=$(pwd)


mkdir -p "${LOGDIR}"
mkdir -p "${GPFSFOLDER}"


export SLURM_EXITCODE=0
export SLURM_STATE="COMPLETED"

###################################################################################
# BENCHMARK After Script ##########################################################
###################################################################################

# set -x
# export TESTNAME="${BENCHMARK}--${DGXSYSTEM}"
# export TESTTYPE="perf-train"
# export LOGNAME="${LOGDIR}/${DATESTAMP}.log"
# export LOGENV="${LOGDIR}/${TESTNAME}_env.json"
# export LOGDLL="${LOGDIR}/${TESTNAME}.json"
# export METADATA_PATH="${CODEDIR}/ci/baselines/${BENCHMARK}/metric_metadata.json"
# export BASELINE_PATH="${CODEDIR}/ci/baselines/${BENCHMARK}/baselines.json"
# python3 ${CODEDIR}/ci/scripts/download_lib_jsons.py --logdir "${LOGDIR}"
# python3 ${CODEDIR}/ci/scripts/mlperf_to_dllog.py --file "${LOGNAME}" --logdir "${LOGDIR}" --env_json_path "${LOGENV}" --dll_path "${LOGDLL}" --benchmark "${BENCHMARK}" --system_config "${DGXSYSTEM}"
# # cat ${LOGENV}
# # cat ${LOGDLL}
# cd ${CODEDIR}
# if [ "${SLURM_EXITCODE}" == "0" ] && [ "${SLURM_STATE}" == "COMPLETED" ]; then
#     python3 -m ci.scripts.joc-ci-tools.log_ops.baseline \
#         "${TESTNAME}" \
#         "${TESTTYPE}" \
#         "${CI_COMMIT_REF_NAME}" \
#         "${LOGDLL}" \
#         "${METADATA_PATH}" \
#         "${BASELINE_PATH}" \
#         --elastic_search_connection ${ES_CONNECTION} \
#         --output_path "${LOGDIR}/${TESTNAME}_baseline.json" \
#         && export TEST_PASS=OK || export TEST_PASS=FAIL;
#     python3 -m ci.scripts.joc-ci-tools.log_ops.save_ci_env > "${LOGDIR}/${TESTNAME}_ci_env.json";
#     python3 -m ci.scripts.joc-ci-tools.log_ops.convert \
#         --log "${LOGDLL}" \
#         --ci_env "${LOGDIR}/${TESTNAME}_ci_env.json" \
#         --fwk_env "${LOGENV}" \
#         --meta "${METADATA_PATH}" \
#         --baseline "${LOGDIR}/${TESTNAME}_baseline.json" \
#         --output "${LOGDIR}/${TESTNAME}_nvdf.json";
# else
#     if [ "${SLURM_STATE}" == "TIMEOUT" ]; then
#         export TEST_PASS=SLURM_TIMEOUT;
#     else
#         export TEST_PASS=SLURM_FAIL;
#     fi;
#     python3 -m ci.scripts.joc-ci-tools.log_ops.save_ci_env > "${LOGDIR}/${TESTNAME}_ci_env.json"
#     python3 -m ci.scripts.joc-ci-tools.log_ops.convert.py \
#         --ci_env "${LOGDIR}/${TESTNAME}_ci_env.json" \
#         --meta "${METADATA_PATH}" \
#         --output "${LOGDIR}/${TESTNAME}_nvdf.json";
# fi
# # cat "${LOGDIR}/${TESTNAME}_ci_env.json"
# # cat "${LOGDIR}/${TESTNAME}_nvdf.json"

# if [[ "${SAVE_TO_KIBANA}" = 1 ]] ; then
#     echo Saving to Kibana;
#     if [[ "${ES_CONNECTION}" = 1 ]] ; then python3 -m ci.scripts.joc-ci-tools.es_tools.post "${LOGDIR}/${TESTNAME}_nvdf.json"; fi;
#     if [[ "${ES_CONNECTION}" = 0 ]] ; then
#         export LOGDEST="${GPFSFOLDER}/PL_${CI_PIPELINE_ID}/${DGXSYSTEM}/${BENCHMARK}${FW_VARIANCE}" && echo "LOGDEST=${LOGDEST}";
#         mkdir -p ${LOGDEST};
#         cp "${LOGDIR}/${TESTNAME}_nvdf.json" ${LOGDEST}/.;
#     fi;
# fi
# echo "TEST_PASS=${TEST_PASS}"

###################################################################################
# SAVE LOGS #######################################################################
###################################################################################

# export DGXSYSTEM="${CI_JOB_NAME#*--}" && export DGXSYSTEM="${DGXSYSTEM%%--*}" && echo "DGXSYSTEM=${DGXSYSTEM}"
# export RESULTSREPO="${GPFSFOLDER}/NVDF" && echo "RESULTSREPO=${RESULTSREPO}"
# git config --global user.name "local"
# git config --global user.email "test@test.com"
# module load git-lfs/2.7.2 || true
# git lfs install || true
# rm -rf "${RESULTSREPO}"
# git clone -b master --single-branch --depth=1 "ssh://git@gitlab-master.nvidia.com:12051/rosluo/NVDF_Results.git" "${RESULTSREPO}"  || true
# cd ${RESULTSREPO}
# git checkout master
# git fetch origin
# git lfs fetch origin || true
# git reset --hard origin/master
# git clean -f
# mkdir -p ${CI_PIPELINE_ID}
# cp -r ${GPFSFOLDER}/PL_${CI_PIPELINE_ID}/* ${CI_PIPELINE_ID}/.
# git lfs track "**/*.json" || true
# git add . || true
# git commit -m "PIPELINE ${CI_PIPELINE_ID}, DGXSYSTEM ${DGXSYSTEM}" || true
# git push || true
# git lfs push origin master|| true
# cd ${GPFSFOLDER}

# individual jobs
# LOGDIR/*.json

# all logs for pipeline
# #GPFS/PL_${PIPELINE_ID}/${DGXSYSTEM}/${BENCHMARK}${FW_VARIANCE}

# all logs for pipeline in NVDF git repo
# GPFS/NVDF/{CI_PIPELINE_ID}/${DGXSYSTEM}/${BENCHMARK}${FW_VARIANCE}


###################################################################################
# UPLOAD LOGS #####################################################################
###################################################################################

# cd ${CODEDIR}
# export RESULTSREPO="${GPFSFOLDER}/NVDF" && echo "RESULTSREPO=${RESULTSREPO}"
# rm -rf "${RESULTSREPO}"
# git clone -b master --single-branch --depth=1 "ssh://git@gitlab-master.nvidia.com:12051/rosluo/NVDF_Results.git" "${RESULTSREPO}"
# if [[ "${SAVE_TO_KIBANA}" = 1 ]]; then
#     echo Saving to Kibana;
#     if [[ -d "${RESULTSREPO}/${CI_PIPELINE_ID}" ]] ; then
#       for log in "${RESULTSREPO}/${CI_PIPELINE_ID}"/*/*/*_nvdf.json; do
#         echo $log;
#         python3 -m ci.scripts.joc-ci-tools.es_tools.post $log;
#       done;
#     fi;
# fi;

# ###################################################################################
# # Generate Reports ################################################################
# ###################################################################################

export CI_PIPELINE_ID=1691500
export CI_PROJECT_ID=9530
export CI_PROJECT_NAME=optimized

cd ~
export REPORTREPO="${PWD}/reports"
module load git-lfs/2.7.2 || true
git lfs install || true
rm -rf "${REPORTREPO}"
git clone -b master --single-branch --depth=1 "ssh://git@gitlab-master.nvidia.com:12051/rosluo/mlperf_reports.git" "${REPORTREPO}" || true
cd ${REPORTREPO}
git checkout master
git fetch origin
git lfs fetch origin || true
git reset --hard origin/master
git clean -f
sudo docker pull gitlab-master.nvidia.com/dl/devops/mlperf-results-scripts:joc_integration
sudo docker run --init \
    -e CI_BUILD_NAME="$CI_BUILD_NAME" \
    -e GITLAB_USER_EMAIL="$GITLAB_USER_EMAIL" \
    -e CI_PIPELINE_ID="$CI_PIPELINE_ID" \
    -e CI_PROJECT_ID="${CI_PROJECT_ID}" \
    -e CI_PROJECT_NAME="${CI_PROJECT_NAME}" \
    -e CI_PROJECT_PATH="${CI_PROJECT_PATH}" \
    --rm \
    --user $(id -u):$(id -g) \
    --mount type=bind,source="${PWD}",target=/workspace/reports \
    gitlab-master.nvidia.com/dl/devops/mlperf-results-scripts:joc_integration /bin/bash \
    -c "bash report.sh"
git lfs track "**/*.html" || true
git add . || true
git commit -m "PIPELINE ${CI_PIPELINE_ID}" || true
git push || true
git lfs push origin master|| true
cd ~
rm -rf "${REPORTREPO}"

# #####

export CI_PIPELINE_ID=1691500
export REPORTREPO=/workspace/reports
export CI_PROJECT_ID=9530
export CI_PROJECT_NAME=optimized
export REPORTDEST=/workspace/reports/public
mkdir -p ${REPORTDEST}

python3 -m joc-ci-tools.pipe_tools.slack_report \
    "xoxb-4916860785-1419760302050-rknNP34tDZosEkbfMspVBtVD" \
    dl/mlperf/optimized \
    ${CI_PIPELINE_ID} \
    ${REPORTDEST} \
    --nightly 1
