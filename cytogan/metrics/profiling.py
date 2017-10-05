import numpy as np
import pandas as pd
import scipy.linalg
import sklearn.metrics.pairwise

from cytogan.extra import logs

log = logs.get_logger(__name__)


def get_nearest_neighbors(examples, neighbors):
    assert len(examples) > 0
    assert len(neighbors) > 0
    example_matrix = np.array(list(examples))
    neighbor_matrix = np.array(list(neighbors))
    # Gives us |examples| x |neighbors| matrix.
    distances = sklearn.metrics.pairwise.cosine_distances(
        example_matrix, neighbor_matrix)
    # Get the indices of the nearest neighbor for each test sample.
    return np.argmin(distances, axis=1)


def reduce_profiles_across_treatments(dataset):
    keys = ('compound', 'concentration', 'moa')
    reduced_profiles = []
    for key, group in dataset.groupby(keys, sort=False, as_index=False):
        mean_profile = group['profile'].mean()
        reduced_profiles.append(key + (mean_profile, ))
    return pd.DataFrame(reduced_profiles, columns=keys + ('profile', ))


def get_whitening_transform(X, epsilon, rotate=True):
    C = (1.0 / X.shape[0]) * np.dot(X.T, X)
    s, V = scipy.linalg.eigh(C)
    D = np.diag(1.0 / np.sqrt(s + epsilon))
    W = np.dot(V, D)
    if rotate:
        W = np.dot(W, V.T)
    return W


def whiten(dataset):
    controls = dataset[dataset['compound'] == 'DMSO']
    mean_control = controls['profile'].mean()
    centered_controls = controls['profile'] - mean_control
    W = get_whitening_transform(centered_controls, epsilon=1e-6, rotate=False)
    dataset['profile'] = np.dot(dataset['profile'] - mean_control, W)


# Performs leave-one-compound-out cross-validation
# Result is a dataframe with columns:
# - image key
# - compound
# - concentration
# - profile vector
# - MOA
def score_profiles(dataset):
    accuracies = []
    # The DMSO should not participate in the MOA classification.
    dataset = dataset[dataset['compound'] != 'DMSO']
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
        if training_data.empty:
            continue

        neighbor_indices = get_nearest_neighbors(test_data['profile'],
                                                 training_data['profile'])
        # Get the MOAs of those nearest neighbors as our predictions.
        predicted_labels = np.array(
            training_data['moa'].iloc[neighbor_indices])
        actual_labels = np.array(test_data['moa'])
        assert actual_labels.shape == predicted_labels.shape
        accuracy = np.mean(predicted_labels == actual_labels)
        log.info('Accuracy for %s is %.3f', holdout_compound, accuracy)
        accuracies.append(accuracy)

        confusion_matrix.loc[actual_labels, predicted_labels] += 1

    return confusion_matrix, np.mean(accuracies)
