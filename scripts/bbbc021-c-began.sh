#!/bin/bash

python3 -m cytogan.train.bbbc021                          \
  --epochs 30                                             \
  --model began                                           \
  --lr 1e-5 1e-5                                          \
  --lr-decay 0.9999                                       \
  --lr-decay-steps 1000                                   \
  --batch-size 16                                         \
  --workspace /data1/peter/runs                           \
  --checkpoint-freq '30s'                                 \
  --summary-freq '30s'                                    \
  --latent-samples 256                                    \
  --generative-samples 10                                 \
  --gpus 0 1 2 3                                          \
  --metadata /data1/peter/metadata/BBBC021_v1_image.csv   \
  --labels /data1/peter/metadata/BBBC021_v1_moa.csv       \
  --images /data1/peter/segmented                         \
  --cell-count-file /data1/peter/metadata/cell_counts.csv \
  --latent-compounds                                      \
  --latent-moa                                            \
  --confusion-matrix                                      \
  --normalize-luminance                                   \
  $@
