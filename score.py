import sklearn.metrics.pairwise
import numpy as np


def get_nearest_neighbors(examples, neighbors):
    print(list(examples), list(neighbors))
    assert len(examples) > 0
    assert len(neighbors) > 0
    example_matrix = np.array(examples)
    neighbor_matrix = np.array(neighbors)
    # Gives us |samples| x |neighbors| matrix.
    distances = sklearn.metrics.pairwise.cosine_distances(
        example_matrix, neighbor_matrix)
    # Get the indices of the nearest neighbor for each test sample.
    return np.argmin(distances, axis=1)


# Performs leave-one-compound-out cross-validation
# Result is a dataframe with columns:
# - image key
# - compound
# - concentration
# - profile vector
# - MOA
def score_profiles(dataset):
    accuracies = []
    unique_labels = len(dataset['moa'].unique())
    confusion_matrix = np.zeros([unique_labels, unique_labels])
    for holdout_compound in dataset['compound']:
        test_mask = dataset['compound'] == holdout_compound
        test_data = dataset[test_mask]
        training_data = dataset[~test_mask]
        if training_data.empty:
            continue

        neighbor_indices = get_nearest_neighbors(test_data['profile'],
                                                 training_data['profile'])

        # Get the MOAs of those nearest neighbors as our predictions.
        predicted_labels = training_data['moa'][neighbor_indices]
        actual_labels = test_data['moa']
        assert actual_labels.shape == predicted_labels.shape
        # Check where the prediction equals the label, and sum over the
        # resulting binary values to get the accuracy.
        accuracy = np.mean(predicted_labels == actual_labels)
        accuracies.append(accuracy)

        confusion_matrix[actual_labels, predicted_labels] += 1

    return confusion_matrix, np.mean(accuracies)
