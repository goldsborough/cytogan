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
- LSGAN: 68%
- LSGAN + Whitening: 76%
- BEGAN: 56%
- C-BEGAN (both): 66%
- C-BEGAN (concentration only but with embedding (mistake)): 68%
- C-LSGAN: 55%
- C-LSGAN (embedding): 53%
- C-LSGAN (no embedding): 48%
- WGAN: 56%
- C-WGAN (concentration only): 48%
- C-WGAN (both): 51%

InfoGan still crap. Collapses immediately. Will try sigmoid + BCE instead of LL
for continuous variables. Also, it seems discriminator wins too early. Will try
lower learning rate for discriminator.

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

### Monday, 10/09/2017

- gpu-vm-1:
 - Continue BEGANS [I am a stupid fucking idiot]
- gpu-vm-2:
  - InfoGAN with 1e-5 8e-5 8e-5 and 2 conditional variables [good]
  - InfoGAN with 7e-5 2e-4 2e-4 and 1 conditional variables
- gpu-vm-3:
 - Continue BEGAN with 0.25 diversity [no noticeable improvement over other BEGAN]
 - WGAN-GP with 8e-5 8e-5 [good, but not better than LSGAN]
- gpu-vm-5:
 - Continue LSGANs
 - Conditional WGAN-GP with 8e-5 8e-5 and embedded conditioning [mistake]

### Tuesday, 10/10/2017

Todo:
[x] Investigate and visualize the latent space of a well-trained LSGAN,
[x] Interpolate between two points and see smoothness of the images,
[x] See if we have control over an InfoGAN via the continuous variables,
[ ] Use only the concentration as condition variable and see if there is a noticeable difference in generated images and latent space,
[x] Subtract the latent vectors between two concentrations of the same compound and see if that encodes the difference in concentration, then apply to another compound and see if that goes to the higher concentration.
[x] Constrained GAN
[x] BIGAN?

### Wednesday, 10/11/2017

What I need:

- Well-trained conditional LSGAN with only the concentration
- Well-trained conditional LSGAN with both
- Well-trained conditional WGAN with only the concentration
- Well-trained conditional WGAN with both
- Well-trained conditional BEGAN with only the concentration
- Well-trained conditional BEGAN with both
- Constrained LSGAN

Launched:

- gpu-vm-1:
 - LSGAN with both
 - LSGAN with only concentration
- gpu-vm-2:
  - C-BEGAN with both
  - C-BEGAN with only concentration
- gpu-vm-3:
  - WGAN with both [fail, retry with smaller lr]
  - WGAN with only concentration [fail, retry with smaller lr]

## Major Findings

- Evaluation of different model architectures
- Quality of generated images
- MOA classification problem and results, compared to VAEs
- Smoothness in latent space
- MOA and compound clustering in latent space
- Conditional GAN results w.r.t. clustering, especially with only concentration
- Vector Algebra?
- InfoGAN and the latent variable
- Whitening
- Cellprofiler features?
- Training tricks I used
- Turing Test
