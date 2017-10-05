from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import collections

def namedtuple(*args, **kwargs):
    T = collections.namedtuple(*args, **kwargs)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    return T

def namedtuple_to_string(named_tuple):
    strings = ['{0} = {1}'.format(*i) for i in named_tuple._asdict().items()]
    return '\n'.join(strings)
