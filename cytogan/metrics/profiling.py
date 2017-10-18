import re

import numpy as np
import pandas as pd
import scipy.linalg
import sklearn.metrics.pairwise

from cytogan.extra import logs

log = logs.get_logger(__name__)


def save_profiles(path, profiles):
    profiles.to_csv(
        path,
        header=True,
        compression='gzip',
        encoding='ascii',
        chunksize=100000)


def load_profiles(path):
    log.info('Loading profiles from %s', path)
    data = pd.read_csv(path, compression='gzip', encoding='ascii')
    parsed_profiles = []
    for p in data['profile']:
        values = re.sub(r'[\[\]\n]', '', p).split()
        parsed_profiles.append(np.array([float(x) for x in values]))
    data.profile = parsed_profiles

    return data


def reduce_profiles_across_treatments(dataset):
    keys = ('compound', 'concentration', 'moa')
    reduced_profiles = []
    for key, group in dataset.groupby(keys, sort=False, as_index=False):
        mean_profile = group['profile'].mean()
        reduced_profiles.append(key + (mean_profile, ))

    reduced = pd.DataFrame(reduced_profiles, columns=keys + ('profile', ))
    reduced.sort_values(['compound', 'concentration'], inplace=True)

    return reduced


def get_whitening_transform(X, epsilon, rotate=True):
    C = (1.0 / X.shape[0]) * np.dot(X.T, X)
    s, V = scipy.linalg.eigh(C)
    D = np.diag(1.0 / np.sqrt(s + epsilon))
    W = np.dot(V, D)
    if rotate:
        W = np.dot(W, V.T)
    return W


def whiten(dataset):
    control_data = dataset[dataset['compound'] == 'DMSO']
    controls = np.array(list(control_data['profile']))
    mean_control = controls.mean(axis=0).reshape(1, -1)
    centered_controls = controls - mean_control

    W = get_whitening_transform(centered_controls, epsilon=1e-6, rotate=False)

    all_profiles = np.array(list(dataset['profile']))
    whitened_profiles = np.dot(all_profiles - mean_control, W)
    dataset['profile'] = list(whitened_profiles)


def log_top_k(compound, distances, test_data, training_data, k=3):
    top_k = np.argsort(distances, axis=1)[:, :k]
    concentrations = test_data.concentration
    moas = test_data.moa
    for i, (c, m, y) in enumerate(zip(concentrations, moas, top_k)):
        top_k_moas = training_data['moa'].iloc[y]
        top_k_strings = []
        for s, j in zip(top_k_moas, y):
            top_k_strings.append('{0} ({1:.8f})'.format(s, distances[i, j]))
        top_k_string = ', '.join(top_k_strings)
        log.info('Top %d neighbors for %s/%.4f (%s): %s'.format(
            k, compound, c, m, top_k_string))


def get_nearest_neighbors(examples, neighbors):
    assert len(examples) > 0
    assert len(neighbors) > 0
    example_matrix = np.array(list(examples))
    neighbor_matrix = np.array(list(neighbors))
    # Gives us |examples| x |neighbors| matrix.
    distances = sklearn.metrics.pairwise.cosine_distances(
        example_matrix, neighbor_matrix)

    # Get the indices of the nearest neighbor for each test sample.
    return distances, np.argmin(distances, axis=1)


def score_profiles(dataset):
    accuracies = []
    labels = dataset['moa'].unique()
    log.info('Have %d MOAs among the profiles.', len(labels))
    confusion_matrix = pd.DataFrame(
        index=labels,
        data=np.zeros([len(labels), len(labels)]),
        columns=labels)
    for holdout_compound in dataset['compound'].unique():
        assert holdout_compound != 'DMSO'
        log.info('Holding out %s', holdout_compound)
        test_mask = dataset['compound'] == holdout_compound
        # Leaves all the concentrations for the holdout compound.
        test_data = dataset[test_mask]
        # All other (compound, concentration) pairs.
        training_data = dataset[~test_mask]

        concentrations_string = ', '.join(map(str, test_data['concentration']))
        log.info(
            'Finding NN for %d concentrations (%s) among %d other treatments',
            len(test_data), concentrations_string, len(training_data))

        distances, nearest = get_nearest_neighbors(test_data['profile'],
                                                   training_data['profile'])

        log_top_k(holdout_compound, distances, test_data, training_data)

        # Get the MOAs of those nearest neighbors as our predictions.
        predicted_labels = np.array(training_data['moa'].iloc[nearest])
        actual_labels = np.array(test_data['moa'])
        assert actual_labels.shape == predicted_labels.shape
        accuracy = np.mean(predicted_labels == actual_labels)
        log.info('Accuracy for %s is %.3f', holdout_compound, accuracy)
        accuracies.append(accuracy)

        for actual in actual_labels:
            for predicted in predicted_labels:
                confusion_matrix[actual][predicted] += 1

    return confusion_matrix, np.mean(accuracies)
