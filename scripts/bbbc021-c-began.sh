#!/bin/bash

python3 -m cytogan.train.bbbc021                          \
  --epochs 30                                             \
  --model c-began                                         \
  --lr 1e-5 1e-5                                          \
  --lr-decay 0.9999                                       \
  --lr-decay-steps 1000                                   \
  --batch-size 16                                         \
  --workspace /data1/peter/runs                           \
  --checkpoint-freq '10min'                               \
  --summary-freq '10min'                                  \
  --generative-samples 10                                 \
  --gpus 2 3                                              \
  --metadata /data1/peter/metadata/BBBC021_v1_image.csv   \
  --labels /data1/peter/metadata/BBBC021_v1_moa.csv       \
  --images /data1/peter/segmented                         \
  --cell-count-file /data1/peter/metadata/cell_counts.csv \
  --latent-compounds                                      \
  --latent-moa                                            \
  --confusion-matrix                                      \
  --normalize-luminance                                   \
  $@
