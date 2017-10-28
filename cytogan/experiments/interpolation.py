import numpy as np

from cytogan.extra import logs

log = logs.get_logger(__name__)


def points_for_treatment(dataset, compound, concentrations, sample_size=None):
    dmso = dataset[dataset['compound'] == 'DMSO']
    if sample_size:
        dmso = dmso.sample(sample_size)

    points = [dmso['profile'].mean(axis=0)]

    compound_index = dataset['compound'] == compound
    for concentration in concentrations:
        concentration_index = dataset['concentration'] == concentration
        treatment = dataset[compound_index & concentration_index]
        if sample_size:
            treatment = treatment.sample(min(len(treatment), sample_size))
        points.append(treatment['profile'].mean(axis=0))

    assert all(p.shape == points[0].shape for p in points), points[1].shape

    return points


def points_from_images(model, cell_data, pool_size=100):
    image_pool = cell_data.next_batch(pool_size)
    indices = np.random.randint(0, pool_size, 2)
    images = np.array(image_pool)[indices]
    start, end = model.encode(images)

    return start, end
