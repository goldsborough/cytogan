import collections

def namedtuple(*args, **kwargs):
    T = collections.namedtuple(*args, **kwargs)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    return T
