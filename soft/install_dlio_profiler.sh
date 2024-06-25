#!/bin/bash
export HWLOC_PREFIX=/soft/libraries/hwloc/
export CPATH=$HWLOC_PREFIX/include/:$CPATH
export LIBRARY_PATH=$HWLOC_PREFIX/lib:$LIBRARY_PATH
pip install cmake
pip uninstall dlio-profiler-py
[ -e /tmp/$USER/dlio-profiler ] || git clone https://github.com/hariharan-devarajan/dlio-profiler.git /tmp/$USER/dlio-profiler
cd /tmp/$USER/dlio-profiler
#git apply $WORKDIR/soft/dlio_profiler_py.patch
export CC=gcc-12
export CXX=g++-12
git submodule update --init --recursive
pip install pybind11 --upgrade
python setup.py bdist_wheel
pip install dist/dlio_profiler_py*.whl --force-reinstall
cp dist/dlio_profiler_py*.whl $WORKDIR/soft/
cd -
