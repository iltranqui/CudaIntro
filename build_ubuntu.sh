#!/bin/bash -e

# fetch vendored deps on a fresh clone (cudnn-frontend enables the fused FP8 forward;
# without it CMake silently builds the slower im2col+cuBLASLt FP8 fallback)
if [ ! -e third_party/cudnn-frontend/include/cudnn_frontend.h ]; then
	echo "Initializing git submodules (cudnn-frontend)..."
	git submodule update --init --depth 1 third_party/cudnn-frontend
fi

mkdir -p build
cd build

# Pick just 1 of the following -- either Release or Debug
set BUILD_TYPE=Release
#set BUILD_TYPE=Debug

cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DENABLE_TIMING_AND_TRACKING=OFF ..
make -j 8
make package

echo Done!
echo Make sure you install the .deb file:
ls -lh *.deb
echo To force the yolo_layer training loss calc onto CPU instead of CUDA, run with:
echo "  DARKNET_YOLO_TRAINING_GPU=cpu darknet detector train ..."
echo "Other values: auto (default), require, verify"

cd ..
