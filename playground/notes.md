# GANs

## Tips

- Train with labels [original GAN paper]
- Use label smoothing (top label is .9 instead of 1.0) [original GAN paper]
- Use strided convolutions instead of max pooling [DCGAN]
- Use batchnorm in all layers except the output layer of G and input of D [DCGAN]
- Remove fully connected layers [DCGAN]
- Use ReLU or Leaky ReLU [DCGAN]
- Use convolutions followed by nearest neighbor resizing for transposed convolutions [distill]
- Consider feature matching for the generator loss [Improved Techniques]
- Minibatch discriminationt to avoid mode collapse [Improved Techniques]
- Use an inception score [Improved Techniques]
- Add Gaussian noise to the input of the discriminator or even to the output of ever D layer
  [Improved Techniques + Inferenc]
- Consider weightnorm as opposed to batchnorm [Improved Techniques]
- Use Wasserstein Loss, but beware of inner loop slowness.

## Personal Observations

- Give discriminator a slightly larger learning rate
