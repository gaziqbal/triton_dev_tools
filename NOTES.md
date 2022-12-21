What tags to use?
I want to use Server release 2.27.0
Corresponds to NGC container 22.10

https://github.com/triton-inference-server/core/tree/r22.11
https://github.com/triton-inference-server/common/tree/r22.10
https://github.com/triton-inference-server/backend/tree/r22.10
https://github.com/triton-inference-server/server/tree/v2.27.0

# Step 1: Create the example model repository 
git clone -b r22.10 https://github.com/triton-inference-server/server.git
cd server/docs/examples
./fetch_models.sh

# Prereqs
sudo apt-get install libboost-all-dev libre2-dev

# ONNXBackend
git clone -b r22.10 https://github.com/triton-inference-server/onnxruntime_backend
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
    -DTRITON_BUILD_ONNXRUNTIME_VERSION=1.13.1 \
    -DTRITON_BUILD_CONTAINER_VERSION=22.10 \
    -DTRITON_BUILD_CONTAINER_VERSION=22.10 \
    -DTRITON_ENABLE_GPU=OFF \
    ..

# Python Backend
git clone -b r22.10 https://github.com/triton-inference-server/python_backend.git
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
    -DTRITON_CORE_REPO_TAG=r22.10 \
    -DTRITON_COMMON_REPO_TAG=r22.10 \
    -DTRITON_BACKEND_REPO_TAG=r22.10 \
    -DTRITON_ENABLE_GPU=OFF \
    ..

# DevTools
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
    -DTRITON_THIRD_PARTY_REPO_TAG=r22.10 \
    -DTRITON_COMMON_REPO_TAG=r22.10 \
    -DTRITON_CORE_HEADERS_ONLY=OFF \
    -DTRITON_ENABLE_GPU=OFF \
    -DTRITON_ENABLE_METRICS_GPU=OFF \
    ..

make --jobs=8 install  

# Runtime

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/giqbal/Source/triton/core/build/install/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/giqbal/Source/triton/core/build/install/lib

# Get site packages
python -m site