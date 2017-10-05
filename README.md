# CytoGAN

# Setting Up Google Cloud

## Installing GCloud

Install `gcloud`: https://cloud.google.com/sdk/downloads
Just grab a version and `wget` it. Then run instructions on the website to install.

Install `gsutil`: `pip install gsutil`.
Authenticate: `gsutil config`.

## Setting up VM

1. Select `us-east1-c` or `us-east1-d` region (have GPUs),
2. For `Machine Type`, click `customize` and set number of GPUs and CPUs,
3. For Disk Type, choose Ubuntu 16.04 with 20 GB SSD drive,
4. Create the instance and wait for it to boot up.
5. SSH into the machine with your browser (easiest).
6. Run the `setup.sh` script found in the `cloud/` folder.

# Notes

- Conditional GAN with a few features (maybe compound categorical)
- Use whitening for features
- Concatenate transfer learning features with GAN features
