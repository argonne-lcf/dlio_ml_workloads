#!/bin/bash
BOOST_DIR=$WORKDIR/soft/boost/1.85.0
LIBAIO_DIR=$WORKDIR//soft/libaio/
export CPATH=$BOOST_DIR/include/:$LIBAIO_DIR/include:$CPATH
export LIBRARY_PATH=$BOOST_DIR/lib:$LIBAIO_DIR/lib:$LIBRARY_PATH
#export CFLAGS="-I${BOOST_LIB} -I${LIBAIO_DIR}/include"
mkdir -p utils/csrc/build
cd utils/csrc/build
cmake .. -DCMAKE_C_FLAGS="-I${BOOST_DIR}/include -I${LIBAIO_DIR}/include" -DCMAKE_CXX_FLAGS="-I${BOOST_DIR}/include -I${LIBAIO_DIR}/include" -DCMAKE_EXE_LINKER_FLAGS="-L${BOOST_DIR}/lib -L${LIBAIO_DIR}/lib -laio"
make -j

/opt/cray/pe/craype/2.7.30/bin/CC -fPIC -O2 -Wall -Wextra -L/soft/applications/conda/2024-04-29/mconda3/lib -Wl,--enable-new-dtags,-rpath,/soft/applications/conda/2024-04-29/mconda3/lib -shared -Wl,-soname,libCosmoflowExt.so -o libCosmoflowExt.so CMakeFiles/CosmoflowExt.dir/src/main.cc.o CMakeFiles/CosmoflowExt.dir/src/file_direct.cc.o CMakeFiles/CosmoflowExt.dir/src/aio_handler.cc.o -L${BOOST_DIR}/lib/ -lboost_iostreams  -lboost_python311 -lboost_numpy311  -L${LIBAIO_DIR}/lib -laio
cp libCosmoflowExt.so ../../
