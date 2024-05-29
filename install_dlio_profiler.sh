#!/bin/bash
export HWLOC_PREFIX=/soft/libraries/hwloc/
export CPATH=$HWLOC_PREFIX/include/:$CPATH
export LIBRARY_PATH=$HWLOC_PREFIX/lib:$LIBRARY_PATH
pip install cmake
pip uninstall dlio-profiler-py
[ -e /tmp/dlio-profiler ] || git clone -b v0.0.3 https://github.com/hariharan-devarajan/dlio-profiler.git /tmp/dlio-profiler
cd /tmp/dlio-profiler
libpath=`which python`
libpath=${libpath%/bin/python}/lib/
git submodule update --init --recursive
rm -rvf $libpath/libgocha* $libpath/libdlio_profiler* $libpath/libcpp-logger.so $libpath/libbrahma* build *.egg*
git pull
pip install pybind11 --upgrade
python setup.py install
cd -
