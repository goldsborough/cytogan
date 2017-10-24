import numpy as np


def slerp(start, end, number_of_samples):
    # https://github.com/soumith/dcgan.torch/issues/14
    # Also: https://arxiv.org/pdf/1609.04468.pdf
    fractions = np.linspace(0, 1, number_of_samples)

    unit_start = start / np.linalg.norm(start, axis=-1).reshape(-1, 1)
    unit_end = end / np.linalg.norm(end, axis=-1).reshape(-1, 1)

    np.testing.assert_allclose(np.linalg.norm(unit_start, axis=-1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(unit_end, axis=-1), 1.0)

    dot_products = np.sum(unit_start * unit_end, axis=-1)

    omega = np.arccos(np.clip(dot_products, -1, 1)).reshape(-1, 1)
    omega_sine = np.sin(omega)

    start, end = np.expand_dims(start, -1), np.expand_dims(end, -1)
    if omega_sine.sum() == 0:
        return (1.0 - fractions) * start + fractions * end

    start_mix = np.sin((1.0 - fractions) * omega) / omega_sine
    end_mix = np.sin(fractions * omega) / omega_sine
    return np.expand_dims(start_mix, 1) * start + \
           np.expand_dims(end_mix, 1) * end


a, b = np.random.randn(2, 100)
s = slerp(a, b, 10)
assert s.shape == (1, 100, 10), s.shape

a, b = np.random.randn(2, 3, 100)
s = slerp(a, b, 5)
assert s.shape == (3, 100, 5), s.shape
