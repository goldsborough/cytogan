#!/bin/bash

# Basic environment stuff
sudo apt-get install -y \
  software-properties-common apt-utils build-essential linux-headers-$(uname -r)

# NVIDIA Drivers
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt-get update

# Basic NVIDIA packages
sudo apt-get install -y  nvidia-cuda-toolkit
sudo apt-get install -y nvidia-367

# Install CUDA (make sure it's 8.0)
CUDA_DEB=cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

if ! [[ -d $CUDA_DEB ]]; then
  wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/$CUDA_DEB"
fi

sudo dpkg -i "$CUDA_DEB"
sudo apt-get update
sudo apt-get install -y cuda-8.0

# Install cuDNN

if ! [[ -f /usr/local/cuda/include/cudnn.h ]]; then
  sudo gsutil -m cp gs://cudnn-imaging/cudnn.tar.gz .
  tar -xvzf cudnn.tar.gz
  sudo cp cuda/include/* /usr/local/cuda/include
  sudo cp cuda/lib64/* /usr/local/cuda/lib64

  echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"' >> ~/.bashrc
  echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
fi

# Install basic python environment
sudo apt-get install -y \
  git python3-numpy python3-dev python3-pip python3-wheel ipython python3-tk

# Install work tools
sudo apt-get install -y emacs vim htop

# Install TensorFlow
sudo pip3 install --upgrade pip && sudo pip3 install tensorflow-gpu

# Install Cytogan
if ! [[ -d cytogan ]]; then
 git clone https://github.com/goldsborough/cytogan
fi

cd cytogan
sudo pip3 install -r requirements.txt

# Download the data

# Remove 3GB of cache
sudo apt-get clean

if [[ ! -d /data1/peter/segmented ]]; then
  sudo mkdir -p /data1/peter/segmented
  sudo mkdir -p /data1/peter/metadata
  sudo mkdir -p /data1/peter/runs
  sudo chown -R $(whoami) /data1/peter

  gsutil -m rsync -r gs://bbbc021-metadata /data1/peter/metadata
  gsutil -m rsync -r gs://bbbc021-segmented /data1/peter/segmented
fi
