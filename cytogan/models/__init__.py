from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
MODELS = ['ae', 'conv_ae', 'vae', 'infogan', 'dcgan', 'lsgan', 'wgan', 'began']
for _gan_name in ('dcgan', 'lsgan', 'wgan', 'began'):
    MODELS.append('c-{0}'.format(_gan_name))
