MODELS = ['ae', 'conv_ae', 'vae', 'infogan', 'dcgan', 'lsgan', 'wgan', 'began']
for _gan_name in ('dcgan', 'lsgan', 'wgan', 'began'):
    MODELS.append('c-{0}'.format(_gan_name))
