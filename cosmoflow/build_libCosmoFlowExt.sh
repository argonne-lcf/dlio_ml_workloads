#!/bin/bash
BOOST_DIR=$WORKDIR/soft/boost/1.85.0
cd $BOOST_DIR/lib
# No sure why it always link to 1.82.0 version
[ -e libboost_iostreams.so.1.82.0 ] || ln -s libboost_iostreams.so.1.85.0 libboost_iostreams.so.1.82.0
LIBAIO_DIR=$WORKDIR//soft/libaio/
export CPATH=$BOOST_DIR/include/:$LIBAIO_DIR/include:$CPATH
export LIBRARY_PATH=$BOOST_DIR/lib:$LIBAIO_DIR/lib:$LIBRARY_PATH
#export CFLAGS="-I${BOOST_LIB} -I${LIBAIO_DIR}/include"
mkdir -p utils/csrc/build
cd utils/csrc/build
cmake .. -DCMAKE_C_FLAGS="-I${BOOST_DIR}/include -I${LIBAIO_DIR}/include" -DCMAKE_CXX_FLAGS="-I${BOOST_DIR}/include -I${LIBAIO_DIR}/include" -DCMAKE_EXE_LINKER_FLAGS="-L${BOOST_DIR}/lib -L${LIBAIO_DIR}/lib -laio" -DCMAKE_INSTALL_PREFIX=$WORKDIR/cosmoflow/utils/
make -j
cp libCosmoflowExt.so ../../
cd -

