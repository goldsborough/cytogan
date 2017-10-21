#!/bin/bash

python3 -m cytogan.train.bbbc021                          \
  --epochs 50                                             \
  --model bigan                                           \
  --lr 7e-5 7e-5 7e-5                                     \
  --lr-decay 0.9999                                       \
  --lr-decay-steps 100                                    \
  --batch-size 64                                         \
  --workspace /data1/peter/runs                           \
  --checkpoint-freq '10min'                               \
  --summary-freq '10min'                                  \
  --generative-samples 196                                \
  --gpus 0 1                                              \
  --metadata /data1/peter/metadata/BBBC021_v1_image.csv   \
  --labels /data1/peter/metadata/BBBC021_v1_moa.csv       \
  --images /data1/peter/segmented                         \
  --cell-count-file /data1/peter/metadata/cell_counts.csv \
  --latent-compounds                                      \
  --latent-concentrations                                 \
  --latent-moa                                            \
  --confusion-matrix                                      \
  --normalize-luminance                                   \
  $@
