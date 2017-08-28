import sklearn.metrics.pairwise
import numpy as np
import pandas as pd


def get_nearest_neighbors(examples, neighbors):
    assert len(examples) > 0
    assert len(neighbors) > 0
    example_matrix = np.array(list(examples))
    neighbor_matrix = np.array(list(neighbors))
    # Gives us |samples| x |neighbors| matrix.
    distances = sklearn.metrics.pairwise.cosine_distances(
        example_matrix, neighbor_matrix)
    # Get the indices of the nearest neighbor for each test sample.
    return np.argmin(distances, axis=1)


def update_confusion_matrix(confusion_matrix, label_table, actual_labels,
                            predicted_labels):
    actual_indices = label_table.loc[actual_labels]['index']
    predicted_indices = label_table.loc[predicted_labels]['index']
    for actual, predicted in zip(actual_indices, predicted_indices):
        confusion_matrix[actual, predicted] += 1

def collapse_profiles():
    pass

# Performs leave-one-compound-out cross-validation
# Result is a dataframe with columns:
# - image key
# - compound
# - concentration
# - profile vector
# - MOA
def score_profiles(dataset):
    accuracies = []
    labels = dataset['moa'].unique()
    confusion_matrix = pd.DataFrame(
        index=labels,
        data=np.zeros([len(labels), len(labels)]),
        columns=labels)
    for holdout_compound in dataset['compound'].unique():
        print('Holding out {0} ...'.format(holdout_compound))
        test_mask = dataset['compound'] == holdout_compound
        test_data = dataset[test_mask]
        training_data = dataset[~test_mask]
        if training_data.empty:
            continue

        neighbor_indices = get_nearest_neighbors(test_data['profile'],
                                                 training_data['profile'])

        # Get the MOAs of those nearest neighbors as our predictions.
        predicted_labels = np.array(training_data['moa'][neighbor_indices])
        actual_labels = np.array(test_data['moa'])
        assert actual_labels.shape == predicted_labels.shape
        accuracy = np.mean(predicted_labels == actual_labels)
        print('Accuracy for {0} is {1:.3f}'.format(holdout_compound, accuracy))
        accuracies.append(accuracy)

        confusion_matrix.loc[actual_labels, predicted_labels] += 1

    return confusion_matrix, np.mean(accuracies)
