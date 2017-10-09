# CytoGAN

# Setting Up Google Cloud

## Installing GCloud

Install Google Cloud's CLI, `gcloud`, on your machine: https://cloud.google.com/sdk/downloads
Just grab a version and `wget` it. Then run instructions on the website to install.

Install `gsutil`: `pip install gsutil`.
Authenticate: `gsutil config`.

## Setting up VM

0. On Google Cloud, go to "Compute Engine", then "VM Instances". Click on "Create Instance".
1. Select `us-east1-c` or `us-east1-d` region (have GPUs),
2. For `Machine Type`, click `customize` and set number of GPUs and CPUs,
3. For Disk Type, choose Ubuntu 16.04 with e.g. 40 GB SSD drive,
4. Create the instance and wait for it to boot up,
5. Setup the correct firewall rules on your own machine: `gcloud compute firewall-rules create default-allow-ssh --allow tcp:22`,
6. SSH into the machine with your browser (easiest) or connect from your machine with `gcloud compute ssh --zone <zone> <instance>`.

To set up port forwarding between your local machine and the VM, use these flags for the `gcloud compute ssh` flags:

`--ssh-flag='-L' --ssh-flag='8000:127.0.0.1:6006'`

# Notes

- Conditional GAN with a few features (maybe compound categorical)
- Use whitening for features
- Concatenate transfer learning features with GAN features

- Try to predict the conditioning code from the images and use that loss
- Use only the concentration as conditional
- Feed the conditional into a dense layer shared between G and D
- Use CellProfiler features, but hand pick a few
- Categorical GAN

So far:

- DCGAN: 43%
- LSGAN: 61%
- LSGAN + Whitening: 68%
- BEGAN: Mode Collapse after 2 epochs (low variety): 55%
- C-LSGAN: 38%?
-

InfoGan still crap. Collapses immediately. Will try sigmoid + BCE instead of LL for continuous variables. Also, it seems discriminator wins too early. Will try lower learning rate for discriminator.

### Saturday, 10/07/2017

- gpu-vm-1:
  - Continue BEGAN from old run
  - New conditional BEGAN with conditional embedding
- gpu-vm-2:
  - InfoGan with BCE and 1e-5 8e-5 8e-5
  - InfoGan with 1e-5 8e-5 8e-5
  - InfoGan with 7e-5 2e-4 2e-4
- gpu-vm-3:
 - Conditional LSGAN with 1e-5 7e-5
 - BEGAN with diversity set to 0.25
- gpu-vm-5:
 - LSGAN with 7e-5 2e-4 and no noise in D layers

Notes:
- InfoGAN all crap,
- BEGAN some promising results. Making the diversity factor 0.25 didn't do much,
- Conditional LSGAN doesn't work. Embedding layer seems to have helped, but not much.
- BEGANs should train more.
- Maybe the problem with InfoGANs so far was that I used too many conditional variables.

### Saturday, 10/09/2017

- gpu-vm-1:
 - Continue BEGANS
- gpu-vm-2:
  - InfoGAN with 1e-5 8e-5 8e-5 and 2 conditional variables
  - InfoGAN with 7e-5 2e-4 2e-4 and 1 conditional variables
- gpu-vm-5:
 - Continue LSGANs
