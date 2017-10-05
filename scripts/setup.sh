#!/bin/bash

sudo apt-get install -y \
  software-properties-common apt-utils build-essential linux-headers-$(uname -r)


sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update

sudo apt-get install -y  --no-install-recommends \
  nvidia-367 nvidia-cuda-toolkit nvidia-kernel-common nvidia-driver nvidia-smi

sudo apt-get install -y \
  git python3-numpy python3-dev python3-pip python3-wheel

sudo apt-get install -y emacs vim

pip3 install --upgrade pip && pip3 install tensorflow-gpu ipython

if [[ ! -d cytogan ]]; then
  git clone https://github.com/goldsborough/cytogan
fi

cd cytogan
pip3 install -r requirements.txt
