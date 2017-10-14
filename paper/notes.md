# Paper Notes

## Introduction

- Large amounts of data
- Introduce challenge of morphological profiling, using computer vision to learn salient (?) representations of cells that partition the latent space according to interesting properties (concentration, compound)
- Leads to allowing statistical analysis, prediction of mechanism of action, cures to diseases
- currently done with cellprofiler or transfer learning, which both have disadvantages
- GANs are a recent player in the unsupervised learning landscape. Due to the generality of the adversarial framework, this opens doors to many new approaches that have to learn good representations of images.
- In this work, we show our success in synthesizing realistic looking images, show our results in representation learning and describe future challenges
- We emphasize biological interpretability of learnt features and synthesis process

large amounts of data because of microscopes
enable many new traditional machine learning and statistical approaches as well as novel deep learning appraoches that can leverage these large amounts of data.
both in assisting biologists in their analysis of biological assays (profiling, screening), as well as solving challenges end-to-end, such as MOA prediction.

## Related Work

- Morphological Profiling with cellprofiler features
- Deep Learning transfer learning
- Autoencoders Nick
- BioGANs

## Realistic Image Synthesis

- GANs have proven to produce highly realistic images. Explain basic GAN framework.
- Explain BBBC021 data
- Evaluated a variety of models. LSGAN and BEGAN stable and promising, albeit different qualitatively
- Interpolation in noise space is smooth
- Vector Algebra is possible, both in noise space and vector space
- Algebra makes the results more interpretable

## Representation Learning

- GAN Framework provides multiple opportunities to learn good representations
- We explore two, DCGAN/LSGAN final layer and BEGAN hidden layer
- Results are better than VAE, but worse than mean profiles and transfer learning
- Yet, the GAN framework provides multiple advantages over the others:
  * Less tuning and generally faster than CellProfiler
  * More tuned to data than transfer learning
  * More room for interpretability and biological usefulness than VAEs, through conditioning or things like InfoGAN

## Conclusion

- We have shown that:
  * GANs can synthesize pretty images, with interpretability possible
  * Representations learnt are superior to GANs and open the door for further analysis
