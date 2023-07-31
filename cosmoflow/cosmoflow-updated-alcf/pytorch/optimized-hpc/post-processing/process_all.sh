#!/bin/bash

usage() {
    echo -e "Usage: ${0} <directory>"
    exit
}

# need one argument
[ "$#" -eq "1" ] || usage
# argument is a valid directory
[ -d "${1}" ] || usage

DRYRUN_DIR=${1}

for i in $(find ${DRYRUN_DIR} -maxdepth 2 -mindepth 2 -type d | grep -v _weak | grep cosmoflow); do
    choose-median.sh ${i} 10 1
done

for i in $(find ${DRYRUN_DIR} -maxdepth 2 -mindepth 2 -type d | grep -v _weak | grep deepcam); do
    choose-median.sh ${i} 5 1
done
