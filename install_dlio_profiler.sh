#!/bin/bash
export HWLOC_PREFIX=$HOME/PolarisAT_eagle/pyenvs/hwloc/
export CPATH=$HWLOC_PREFIX/include/:$CPATH
export LIBRARY_PATH=$HWLOC_PREFIX/lib:$LIBRARY_PATH
module load cmake
pip uninstall dlio-profiler-py
[ -e /tmp/dlio-profiler ] || git clone https://github.com/hariharan-devarajan/dlio-profiler.git
git clone https://github.com/open-mpi/hwloc.git
cd hwloc
./autogen.sh
./configure --prefix=$HWLOC_PREFIX
make all install -j4
cd -
cd /tmp/dlio-profiler
libpath=`which python`
libpath=${libpath%/bin/python}/lib/
git submodule update --init --recursive
rm -rvf $libpath/libgocha* $libpath/libdlio_profiler* $libpath/libcpp-logger.so $libpath/libbrahma* build *.egg*
git pull
pip install pybind11 --upgrade
python setup.py install
cd -
