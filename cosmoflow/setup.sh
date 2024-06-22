#!/bin/sh
source ${WORKDIR}/setup_ml_env.sh
BOOST_DIR=${WORKDIR}/soft/boost/1.85.0/
export CPATH=${BOOST_DIR}/:$CPATH
export LD_LIBRARY_PATH=${BOOST_DIR}/stage/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${WORKDIR}/cosmoflow/utils/csrc/build/:$LD_LIBRARY_PATH
