#!/bin/bash

# Basic environment stuff
sudo apt-get install -y \
software-properties-common apt-utils build-essential linux-headers-$(uname -r)

# NVIDIA Drivers
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update

# Basic NVIDIA packages
sudo apt-get install -y  --no-install-recommends \
  nvidia-367 nvidia-cuda-toolkit nvidia-kernel-common nvidia-driver nvidia-smi

# Install CUDA (make sure it's 8.0)
CUDA_DEB=cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

if [[! -d $CUDA_DEB ]]; then
  wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/$CUDA_DEB"
fi

sudo dpkg -i "$CUDA_DEB"
sudo apt-get update
sudo apt-get install cuda-8.0

# Install cuDNN

gsutil -m cp gs://cudnn-imaging/cudnn.tar.gz

# Install basic python environment
sudo apt-get install -y \
  git python3-numpy python3-dev python3-pip python3-wheel ipython

# Install work tools
sudo apt-get install -y emacs vim

# Install TensorFlow
pip3 install --upgrade pip && pip3 install tensorflow-gpu

# Install Cytogan
if [[ ! -d cytogan ]]; then
 git clone https://github.com/goldsborough/cytogan
fi

cd cytogan
pip3 install -r requirements.txt

# Download the data
sudo mkdir -p /data1/peter/segmented
sudo mkdir -p /data1/peter/metadata

gsutil -m rsync -r gs://bbbc021-metadata /data1/peter/metadata
gsutil -m rsync -r gs://bbbc021-segmented /data1/peter/segmented
