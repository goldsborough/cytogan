#!/bin/bash

job_name="cytogan_test_$(date +'%s')"
echo $job_name
gcloud ml-engine jobs submit training $job_name   \
        --stream-logs                             \
        --package-path "cytogan_test"             \
        --module-name "cytogan_test.task"         \
        --job-dir "gs://cytogan-output/$job_name" \
        --region "us-east1"                       \
        --config config.yaml                      \
        -- \
        --image "gs://bbbc021-segmented/Week1_22123/Week1_150607_B02_s1_w107447158-AC76-4844-8431-E6A954BD1174-1.png"
