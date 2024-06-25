#!/bin/bash
mkdir -p /tmp/$USER/
alias gcc='gcc-12'
alias g++='g++-12'
export VERSION=1.85.0
export SRC=https://boostorg.jfrog.io/artifactory/main/release/$VERSION/source/boost_1_85_0.tar.gz
cd /tmp/$USER/
wget $SRC && tar -xzf boost_1_85_0.tar.gz
cd boost_1_85_0
#sed -i "s/TOOLSET=\"\"/TOOLSET=gcc-12/g" bootstrap.sh
./bootstrap.sh --prefix=${WORKDIR}/soft/boost/1.85.0 --with-python=python3
./b2
./b2 install
cd -

