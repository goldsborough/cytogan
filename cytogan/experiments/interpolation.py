import numpy as np


def points_for_treatment(treatment_profiles, compound, concentration):
    com = treatment_profiles['compound'] == compound
    con = treatment_profiles['concentration'] == float(concentration)
    treatment = np.array(treatment_profiles[com & con]['profile'].values)
    print(treatment.shape)
    assert np.ndim(treatment) > 0, treatment.shape

    dmso = treatment_profiles[treatment_profiles['compound'] == 'DMSO']
    dmso = np.array(dmso['profile'].values)

    return dmso, treatment


def points_from_images(model, cell_data, pool_size=100):
    image_pool = cell_data.next_batch(pool_size)
    indices = np.random.randint(0, pool_size, 2)
    images = np.array(image_pool)[indices]
    start, end = model.encode(images)

    return start, end
