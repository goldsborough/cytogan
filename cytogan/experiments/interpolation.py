import numpy as np


def points_for_treatment(dataset, compound, concentration, sample_size=None):
    com = dataset['compound'] == compound
    con = dataset['concentration'] == float(concentration)
    treatment = dataset[com & con]

    if sample_size:
        treatment = treatment.sample(sample_size)

    treatment_vector = treatment['profile'].mean(axis=0)

    dmso = dataset[dataset['compound'] == 'DMSO']
    dmso_vector = dmso['profile'].mean(axis=0)

    return dmso_vector, treatment_vector


def points_from_images(model, cell_data, pool_size=100):
    image_pool = cell_data.next_batch(pool_size)
    indices = np.random.randint(0, pool_size, 2)
    images = np.array(image_pool)[indices]
    start, end = model.encode(images)

    return start, end
