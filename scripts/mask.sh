#!/bin/bash

python3 -m scripts.mask \
  --metadata /data1/peter/metadata/BBBC021_v1_image.csv \
  --masks /data1/peter/masks \
  --image-path /data1/peter/images/ \
  --output /data1/peter/segmented/
