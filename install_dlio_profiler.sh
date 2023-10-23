#!/bin/bash
module load cmake
pip uninstall dlio-profiler-py
[ -e dlio-profiler ] || git clone https://github.com/hariharan-devarajan/dlio-profiler.git
cd dlio-profiler
libpath=`which python`
libpath=${libpath%/bin/python}/lib/
git submodule update --init --recursive
rm -rvf $libpath/libgocha* $libpath/libdlio_profiler* $libpath/libcpp-logger.so $libpath/libbrahma*
git pull
pip install pybind11 --upgrade
rm -rvf build *.egg*
python setup.py build
python setup.py install
cd -
