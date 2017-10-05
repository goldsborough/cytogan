#!/bin/bash

sudo apt-get install -y \
  software-properties-common apt-utils build-essential

sudo apt-get install -y \
  git python3-numpy python3-dev python3-pip python3-wheel

sudo apt-get install -y emacs vim

pip3 install --upgrade pip && pip3 install tensorflow-gpu

git clone https://github.com/goldsborough/cytogan

cd cytogan

pip3 install -r requirements.txt
