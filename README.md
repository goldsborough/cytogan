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
4. Create the instance and wait for it to boot up,
5. Setup the correct firewall rules on your own machine: `gcloud compute firewall-rules create default-allow-ssh --allow tcp:22`,
6. SSH into the machine with your browser (easiest) or connect from your machine with `gcloud compute ssh --zone <zone> <instance>`,
7. Run the `setup.sh` script found in the `cloud/` folder.

To set up port forwarding between your local machine and the VM, use these flags for the `gcloud compute ssh` flags:

`--ssh-flag='-L' --ssh-flag='8000:127.0.0.1:6006'`

# Notes

- Conditional GAN with a few features (maybe compound categorical)
- Use whitening for features
- Concatenate transfer learning features with GAN features

So far:

- DCGAN: 43%
- LSGAN: 61%
- LSGAN + Whitening: 68%
