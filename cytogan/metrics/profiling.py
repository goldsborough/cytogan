import numpy as np
import pandas as pd
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


def _reduce_profiles_across_compounds(dataset):
    keys = ('compound', 'concentration', 'moa')
    reduced_profiles = []
    for key, group in dataset.groupby(keys, sort=False, as_index=False):
        mean_profile = group['profile'].mean()
        reduced_profiles.append(key + (mean_profile, ))
    return pd.DataFrame(reduced_profiles, columns=keys + ('profile', ))


# Performs leave-one-compound-out cross-validation
# Result is a dataframe with columns:
# - image key
# - compound
# - concentration
# - profile vector
# - MOA
def score_profiles(full_dataset):
    accuracies = []
    dataset = _reduce_profiles_across_compounds(full_dataset)
    log.info('Reduced dataset from %d to %d profiles for each '
             '(compound, concentration) pair ...',
             len(full_dataset), len(dataset))
    labels = dataset['moa'].unique()
    log.info('Have %d MOAs among the profiles.', len(labels))
    confusion_matrix = pd.DataFrame(
        index=labels,
        data=np.zeros([len(labels), len(labels)]),
        columns=labels)
    for holdout_compound in dataset['compound'].unique():
        log.info('Holding out %d ...', holdout_compound)
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
        log.info('Accuracy for %s is %.3f'.format(holdout_compound, accuracy))
        accuracies.append(accuracy)

        confusion_matrix.loc[actual_labels, predicted_labels] += 1

    return confusion_matrix, np.mean(accuracies)
