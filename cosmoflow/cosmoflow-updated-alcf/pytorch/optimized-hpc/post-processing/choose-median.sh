#!/bin/bash

usage() {
    echo -e "Usage: ${0} <directory> <N> <T>"
    echo -e "\twhere N > 0 is the required number of runs to produce a result"
    echo -e "\tand 0 < T < N/2 is the number of runs to trim from top and bottom of the trimmed mean"
    exit
}
# need three argument
[ "$#" -eq "3" ] || usage
# first arg is an existing directory
[ -d "${1}" ] || usage
# trick to test if a string is a valid integer
[ "${2}" -eq "${2}" ] || usage
[ "${3}" -eq "${3}" ] || usage

LOG_DIRECTORY=${1}
MLPERF_BENCHMARK_N=${2}
MLPERF_BENCHMARK_T=${3}

# N > 0
[ "${MLPERF_BENCHMARK_N}" -gt "0" ] || usage
# T >= 0
[ "${MLPERF_BENCHMARK_T}" -ge "0" ] || usage
# N-2T > 0
[ "$(( ${MLPERF_BENCHMARK_N} - (2 * ${MLPERF_BENCHMARK_T}) ))" -ge "1" ] || usage

#########################################################################
# make sure tmpfiles gets deleted when script exits (except for kill -9)
#########################################################################
on_exit() {
    rm -rf ${MYTMPFILE}
    rm -rf ${MYWINDOLIST}
}
trap on_exit EXIT

MYTMPFILE="$(mktemp)"
MYWINDOWLIST="$(mktemp)"


#########################################################################
# move into directory to process
#########################################################################
cd $1


#########################################################################
# get the appropriate files and sort them numerically and lexicographically
#########################################################################
for i in $(find . -maxdepth 1 -name 'run_*_*.log' | sort -t_ -n -k2,2 -k3,3 -k4,4 -k5,5); do
    runtime=$(mllog-score.py $i)
    if [[ "${runtime}" ]]; then
	echo -e ${i}'\t'${runtime}
	# if python -m mlperf_logging.compliance_checker --hpc blah blah > ${MYTMPCOMPLIANCEOUT}; then
	#       else
	# >&2 echo compliance failed for ${i}:
	# >&2 cat ${MYTMPCOMPLIANCEOUT}
    else
	>&2 echo ${i} has no run_start or run_stop tags
    fi
done > ${MYTMPFILE}

NUM_RESULTS=$(wc --lines ${MYTMPFILE} | tr -s ' ' '\t' | cut -f1)
echo "there are ${NUM_RESULTS} results in ${1}"

#########################################################################
# helper functions
#########################################################################
slice_window() {
    # first arg is start line
    # second arg is window size
    # stdin is N lines of input
    # stdout is window_size lines of file from [start_line, start_line+window_size)
    tail --lines=+${1} | head --lines=${2}
}

trim_window() {
    # first arg is number of lines to trim from top and bottom
    head --lines=-${1} | tail --lines=+$(( ${1} + 1 ))
}

average() {
    # sum lines and divide by number of records ("NR")
    awk '{s+=$1}END{print s/NR}'
}

score_window() {
    # first arg is the start line for the window
    # second arg is window size
    # third arg is amount to trim from top and bottom
    # stdin is list of run scores
    slice_window ${1} ${2} | sort -n | trim_window ${3} | average
}

NUM_WINDOWS=$(( ${NUM_RESULTS} + 1 - ${MLPERF_BENCHMARK_N} ))

#########################################################################
# get the scores for all the windows
#########################################################################
echo -e "window\tscore"
for i in $(seq 1 ${NUM_WINDOWS}); do
    echo -ne ${i}'\t'; cut -f2 ${MYTMPFILE} | score_window $i ${MLPERF_BENCHMARK_N} ${MLPERF_BENCHMARK_T}
done | sort -n -k2,2 | tee ${MYWINDOWLIST}

#########################################################################
# find the median of the windows, if there are an even number of windows use
# the worse of the scores nearest the center of the list
#########################################################################
MEDIAN_WINDOW=$(tail -n+$(( ${NUM_WINDOWS} / 2 + 1 )) ${MYWINDOWLIST} | head -n1 | cut -f1)
MEDIAN_SCORE=$(cut -f2 ${MYTMPFILE} | score_window ${MEDIAN_WINDOW} ${MLPERF_BENCHMARK_N} ${MLPERF_BENCHMARK_T})
echo
echo "median window is ${MEDIAN_WINDOW} with score of ${MEDIAN_SCORE}"

echo using the following results
slice_window ${MEDIAN_WINDOW} ${MLPERF_BENCHMARK_N} < ${MYTMPFILE} | awk -v OFS='\t' '{print $0, "result_"NR".txt"}' | tee ${MYWINDOWLIST}

#########################################################################
# copy all the result files to their "official" names
#########################################################################
awk '{print "cp", $1, $3}' ${MYWINDOWLIST} | bash
