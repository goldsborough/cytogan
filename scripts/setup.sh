#!/bin/bash

sudo apt-get install -y \
  software-properties-common apt-utils build-essential linux-headers-$(uname -r)

sudo apt-get install -y  --no-install-recommends \
  nvidia-cuda-toolkit nvidia-kernel-common nvidia-driver nvidia-smi

sudo apt-get install -y \
  git python3-numpy python3-dev python3-pip python3-wheel

sudo apt-get install -y emacs vim

# Create a system wide NVBLAS config
# See http://docs.nvidia.com/cuda/nvblas/
NVBLAS_CONFIG_FILE=/etc/nvidia/nvblas.conf
cat << EOF >> ${NVBLAS_CONFIG_FILE}
# Insert here the CPU BLAS fallback library of your choice.
# The standard libblas.so.3 defaults to OpenBLAS, which does not have the
# requisite CBLAS API.
NVBLAS_CPU_BLAS_LIB /usr/lib/libblas/libblas.so

# Use all GPUs
NVBLAS_GPU_LIST ALL

pip3 install --upgrade pip && pip3 install tensorflow-gpu ipython

git clone https://github.com/goldsborough/cytogan

cd cytogan

pip3 install -r requirements.txt
