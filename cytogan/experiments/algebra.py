import abc

import numpy as np

from cytogan.extra import logs
from cytogan.metrics import profiling

log = logs.get_logger(__name__)


def select_top_k(treatment_profiles, nearest_neighbors, columns):
    count_map = {n: 0 for n in nearest_neighbors}
    for neighbor in nearest_neighbors:
        count_map[neighbor] += 1
    counts = list(count_map.items())
    counts.sort(key=lambda p: p[1], reverse=True)
    top_k_indices, top_k_counts = list(zip(*counts[:3]))
    top_k = treatment_profiles[columns].iloc[list(top_k_indices)]

    return top_k, top_k_counts


class Experiment(abc.ABC):
    def __init__(self, number_of_experiments):
        self.size = None
        self.number_of_experiments = number_of_experiments

    @abc.abstractmethod
    def keys(self, cell_data, amount):
        pass

    @abc.abstractmethod
    def evaluate(self, result_vectors, treatment_profiles):
        pass

    def calculate(self, model, lhs, rhs, base):
        assert len(lhs) == len(rhs) == len(base) == self.size, (len(lhs),
                                                                len(rhs),
                                                                len(base),
                                                                self.size)
        images = np.concatenate([lhs, rhs, base], axis=0)
        vectors = model.encode(images.squeeze())

        lhs, rhs, base = np.split(vectors, 3, axis=0)
        result = (lhs - rhs) + base
        images = model.generate(result)
        assert result.shape == (self.size, model.latent_size), result.shape
        assert len(result) == len(images) == len(lhs) == self.size, (
            len(result), len(images))
        vectors = np.concatenate([lhs, rhs, base, result], axis=0)

        return vectors, images

    def constrain_size(self, lhs, rhs, base, maximum_amount):
        log.info('Have %d lhs, %d rhs, %d base samples available for %s',
                 len(lhs), len(rhs), len(base), self.name)

        total_amount = self.number_of_experiments * maximum_amount
        maximum_amount = min(len(lhs), len(rhs), len(base), total_amount)
        lhs = lhs.sample(maximum_amount)
        rhs = rhs.sample(maximum_amount)
        base = base.sample(maximum_amount)
        assert len(lhs) == len(rhs) == len(base), (len(lhs), len(rhs),
                                                   len(base))
        log.info('Using %d (lhs, rhs, base) pairs for %d %s experiments',
                 len(lhs), self.number_of_experiments, self.name)

        return lhs, rhs, base, maximum_amount


class MoaCanceling(Experiment):
    def __init__(self, number_of_experiments):
        super(MoaCanceling, self).__init__(number_of_experiments)
        self.name = 'MOA canceling'
        self.compound = 'emetine'
        self.concentration = 1.0

    def keys(self, cell_data, maximum_amount):
        com = cell_data.metadata['compound'] == self.compound
        con = cell_data.metadata['concentration'] == self.concentration
        lhs = cell_data.metadata[com & con]
        assert len(lhs) > 0

        dmso = cell_data.metadata[cell_data.metadata['compound'] == 'DMSO']
        rhs = dmso[:len(dmso) // 2]
        base = dmso[len(dmso) // 2:]

        constrained = self.constrain_size(lhs, rhs, base, maximum_amount)
        lhs, rhs, base, self.size = constrained

        return list(np.concatenate([lhs.index, rhs.index, base.index]))

    def evaluate(self, result_vectors, treatment_profiles):
        assert len(result_vectors) == self.size
        groups = np.split(result_vectors, self.number_of_experiments, axis=0)
        mean_result_vectors = np.array([g.mean(axis=0) for g in groups])
        _, nearest_neighbors = profiling.get_nearest_neighbors(
            mean_result_vectors, treatment_profiles['profile'])
        moas = np.array(treatment_profiles['moa'].iloc[nearest_neighbors])

        com = treatment_profiles['compound'] == self.compound
        con = treatment_profiles['concentration'] == self.concentration
        target_moa = treatment_profiles[com & con].iloc[0]['moa']
        log.info('Target MOA for MOA canceling experiment is: %s', target_moa)

        accuracy = np.mean(moas == target_moa)
        log.info('Accuracy for MOA canceling experiment: %.3f', accuracy)

        top_k_moas, top_k_counts = select_top_k(treatment_profiles,
                                                nearest_neighbors, 'moa')
        top_k_pairs = ((m, str(c)) for m, c in zip(top_k_moas, top_k_counts))
        top_k_string = ', '.join('{} ({})'.format(*i) for i in top_k_pairs)

        log.info('Top MOAs for MOA canceling experiment (correct: %s): %s',
                 target_moa, top_k_string)

        lhs = np.expand_dims([target_moa] * len(moas), axis=1)
        rhs = base = np.expand_dims(['DMSO'] * len(lhs), axis=1)
        moas = np.expand_dims(moas, axis=1)
        labels = np.concatenate([lhs, rhs, base, moas], axis=1)

        return labels


class ConcentrationDistance(Experiment):
    def __init__(self, number_of_experiments):
        super(ConcentrationDistance, self).__init__(number_of_experiments)
        self.name = 'Concentration Distance'
        self.start_compound = 'emetine'
        self.start_concentration = 0.1
        self.target_compound = 'ALLN'
        self.target_concentration = 1.0
        self.lhs_treatment = '{0}/{1}'.format(self.start_compound,
                                              self.target_concentration)
        self.rhs_treatment = '{0}/{1}'.format(self.start_compound,
                                              self.start_concentration)
        self.base_treatment = '{0}/{1}'.format(self.target_compound,
                                               self.start_concentration)

    def keys(self, cell_data, maximum_amount):
        com = cell_data.metadata['compound'] == self.start_compound
        con = cell_data.metadata['concentration'] == self.target_concentration
        lhs = cell_data.metadata[com & con]
        assert len(lhs) > 0

        con = cell_data.metadata['concentration'] == self.start_concentration
        rhs = cell_data.metadata[com & con]
        assert len(rhs) > 0

        com = cell_data.metadata['compound'] == self.target_compound
        con = cell_data.metadata['concentration'] == self.start_concentration
        base = cell_data.metadata[com & con]
        assert len(rhs) > 0

        constrained = self.constrain_size(lhs, rhs, base, maximum_amount)
        lhs, rhs, base, self.size = constrained

        return list(np.concatenate([lhs.index, rhs.index, base.index]))

    def evaluate(self, result_vectors, treatment_profiles):
        assert len(result_vectors) == self.size
        groups = np.split(result_vectors, self.number_of_experiments, axis=0)
        mean_result_vectors = np.array([g.mean(axis=0) for g in groups])
        _, nearest_neighbors = profiling.get_nearest_neighbors(
            mean_result_vectors, treatment_profiles['profile'])

        result = treatment_profiles[['compound',
                                     'concentration']].iloc[nearest_neighbors]

        com = result['compound'] == self.target_compound
        con = result['concentration'] == self.target_concentration
        correct = result[com & con]

        accuracy = len(correct) / len(result)
        log.info('Accuracy for %s experiment: %.3f', self.name, accuracy)

        top_k, top_k_counts = select_top_k(treatment_profiles,
                                           nearest_neighbors,
                                           ['compound', 'concentration'])
        top_k = ('{0}/{1}'.format(com, con) for com, con in top_k.values)
        top_k_pairs = ((m, str(c)) for m, c in zip(top_k, top_k_counts))
        top_k_string = ', '.join('{0} ({1})'.format(*i) for i in top_k_pairs)

        log.info('Top treatments for %s experiment (correct: %s/%s): %s',
                 self.name, self.target_compound, self.target_concentration,
                 top_k_string)

        lhs = np.expand_dims([self.lhs_treatment] * len(result), axis=1)
        rhs = np.expand_dims([self.rhs_treatment] * len(result), axis=1)
        base = np.expand_dims([self.base_treatment] * len(result), axis=1)
        result = [['{0}/{1}'.format(com, con)] for com, con in result.values]
        labels = np.concatenate([lhs, rhs, base, result], axis=1)

        return labels


EXPERIMENT_MAP = {
    'moa-canceling': MoaCanceling,
    'concentration-distance': ConcentrationDistance,
}
EXPERIMENTS = list(EXPERIMENT_MAP.keys())


def get_experiment(name, number_of_experiments):
    return EXPERIMENT_MAP[name](number_of_experiments)
