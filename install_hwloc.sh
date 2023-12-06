#!/bin/bash
git clone  https://github.com/open-mpi/hwloc
export HWLOC_PREFIX=$HOME/PolarisAT_eagle/pyenvs/
cd hwloc
./autogen.sh
./configure --prefix=${HWLOC_PREFIX}
make -j all install
export CPATH=$HWLOC_PREFIX/include:$CPATH
export LIBRARY_PATH=$HWLOC_PREFIX/lib:$LIBRARY_PATH


