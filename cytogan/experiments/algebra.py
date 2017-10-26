import abc

import numpy as np

from cytogan.extra import logs
from cytogan.metrics import profiling

log = logs.get_logger(__name__)


class Experiment(abc.ABC):
    @abc.abstractmethod
    def keys(self, cell_data, amount):
        pass

    @abc.abstractmethod
    def evaluate(self, result_vectors, treatment_profiles):
        pass

    def calculate(self, model, lhs, rhs, base):
        assert len(lhs) == len(rhs) == len(base) == self.size, (len(lhs),
                                                                len(rhs),
                                                                len(base))
        images = np.concatenate([lhs, rhs, base], axis=0)
        vectors = model.encode(images.squeeze())

        lhs, rhs, base = np.split(vectors, 3, axis=0)
        print(lhs.shape, rhs.shape, base.shape)
        result = (lhs - rhs) + base
        print(result.shape)
        images = model.generate(result)
        assert result.shape == (self.size, model.latent_size), result.shape
        assert len(result) == len(images) == len(lhs) == self.size, (
            len(result), len(images))
        vectors = np.concatenate([lhs, rhs, base, result], axis=0)
        print(vectors.shape)

        return vectors, images

    def constrain_size(self, lhs, rhs, base, number_of_experiments,
                       maximum_amount):
        log.info('Have %d lhs, %d rhs, %d base samples available for %s',
                 len(lhs), len(rhs), len(base), self.name)

        total_amount = number_of_experiments * maximum_amount
        maximum_amount = min(len(lhs), len(rhs), len(base), total_amount)
        lhs = lhs.sample(maximum_amount)
        rhs = rhs.sample(maximum_amount)
        base = base.sample(maximum_amount)
        assert len(lhs) == len(rhs) == len(base), (len(lhs), len(rhs),
                                                   len(base))
        log.info('Using %d (lhs, rhs, base) pairs for %d %s experiments',
                 len(lhs), number_of_experiments, self.name)

        lhs, rhs, base = np.split(lhs, 3, axis=0)

        return lhs, rhs, base, maximum_amount


class MoaCanceling(Experiment):
    def __init__(self):
        self.name = 'MOA canceling'
        self.compound = 'emetine'
        self.concentration = 1.0
        self.size = None

    def keys(self, cell_data, number_of_experiments, maximum_amount):
        com = cell_data.metadata['compound'] == self.compound
        con = cell_data.metadata['concentration'] == self.concentration
        lhs = cell_data.metadata[com & con]
        assert len(lhs) > 0

        dmso = cell_data.metadata[cell_data.metadata['compound'] == 'DMSO']
        rhs = dmso[:len(dmso) // 2]
        base = dmso[len(dmso) // 2:]

        constrained = self.constrain_size(
            lhs, rhs, base, number_of_experiments, maximum_amount)
        lhs, rhs, base, self.size = constrained

        return list(np.concatenate([lhs.index, rhs.index, base.index]))

    def evaluate(self, result_vectors, treatment_profiles,
                 number_of_equations):
        assert len(result_vectors) == self.size
        groups = np.split(result_vectors, number_of_equations, axis=0)
        mean_result_vectors = np.array([g.mean(axis=0) for g in groups])
        np.savetxt('result.csv', result_vectors, delimiter=',')
        _, nearest_neighbors = profiling.get_nearest_neighbors(
            mean_result_vectors, treatment_profiles['profile'])
        moas = np.array(treatment_profiles['moa'].iloc[nearest_neighbors])
        print(moas)

        com = treatment_profiles['compound'] == self.compound
        con = treatment_profiles['concentration'] == self.concentration
        target_moa = treatment_profiles[com & con].iloc[0]['moa']
        log.info('Target MOA for MOA canceling experiment is: %s', target_moa)

        accuracy = np.mean(moas == target_moa)
        log.info('Accuracy for MOA canceling experiment: %.3f', accuracy)

        count_map = {n: 0 for n in nearest_neighbors}
        for neighbor in nearest_neighbors:
            count_map[neighbor] += 1
        counts = list(count_map.items())
        counts.sort(key=lambda p: p[1], reverse=True)
        top_k_indices, top_k_counts = list(zip(*counts[:3]))
        top_k_moas = treatment_profiles['moa'].iloc[list(top_k_indices)]

        top_k_pairs = ((m, str(c)) for m, c in zip(top_k_moas, top_k_counts))
        top_k_string = ', '.join('{} ({})'.format(*i) for i in top_k_pairs)

        log.info('Top 3 MOAs for MOA canceling experiment (correct: %s): %s',
                 target_moa, top_k_string)

        lhs = np.expand_dims([target_moa] * len(moas), axis=1)
        rhs = base = np.expand_dims(['DMSO'] * len(lhs), axis=1)
        moas = np.expand_dims(moas, axis=1)
        labels = np.concatenate([lhs, rhs, base, moas], axis=1)

        return labels


class ConcentrationDistance(Experiment):
    def __init__(self):
        self.name = 'concentration Distance'

    def keys(self, cell_data, maximum_amount):
        com = cell_data.metadata['compound'] == 'emetine'
        con = cell_data.metadata['concentration'] == 1.0
        lhs = cell_data.metadata[com & con]
        assert len(lhs) > 0

        con = cell_data.metadata['concentration'] == 0.1
        rhs = cell_data.metadata[com & con]
        assert len(rhs) > 0

        com = cell_data.metadata['compound'] == 'ALLN'
        con = cell_data.metadata['concentration'] == 0.1
        base = cell_data.metadata[com & con]
        assert len(rhs) > 0

        lhs, rhs, base = self.constrain_size(lhs, rhs, base, maximum_amount)

        return list(np.concatenate([lhs.index, rhs.index, base.index]))

    def evaluate(self, result_vectors, treatment_profiles):
        pass


EXPERIMENT_MAP = dict(
    moa_canceling=MoaCanceling, concentration_distance=ConcentrationDistance)
EXPERIMENTS = list(EXPERIMENT_MAP.keys())


def get_experiment(name):
    return EXPERIMENT_MAP[name]()
