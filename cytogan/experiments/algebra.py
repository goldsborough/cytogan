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
        images = np.concatenate([lhs, rhs, base], axis=0)
        vectors = model.encode(images.squeeze())

        lhs, rhs, base = np.split(vectors, 3, axis=0)
        result = base + (lhs - rhs)
        images = model.generate(result)
        vectors = np.concatenate([lhs, rhs, base, result])

        return vectors, images

    def constrain_size(self, lhs, rhs, base, maximum_amount):
        log.info('Have %d lhs, %d rhs, %d base samples available for %s',
                 len(lhs), len(rhs), len(base), self.name)

        if maximum_amount:
            maximum_amount = min(len(lhs), len(rhs), len(base), maximum_amount)
            lhs = lhs[:maximum_amount]
            rhs = rhs[:maximum_amount]
            base = base[:maximum_amount]

        assert len(lhs) == len(rhs) == len(base), (len(lhs), len(rhs),
                                                   len(base))
        log.info('Using %d (lhs, rhs, base) pairs for %s experiment',
                 len(lhs), self.name)

        return lhs, rhs, base


class MoaCanceling(Experiment):
    def __init__(self):
        self.name = 'MOA canceling'
        self.compound = 'emetine'

    def keys(self, cell_data, maximum_amount):
        com = cell_data.metadata['compound'] == self.compound
        lhs = cell_data.metadata[com]
        assert len(lhs) > 0

        dmso = cell_data.metadata[cell_data.metadata['compound'] == 'DMSO']
        rhs = dmso[:len(dmso) // 2]
        base = dmso[len(dmso) // 2:]

        lhs, rhs, base = self.constrain_size(lhs, rhs, base, maximum_amount)

        return list(np.concatenate([lhs.index, rhs.index, base.index]))

    def evaluate(self, result_vectors, treatment_profiles):
        _, nearest_neighbors = profiling.get_nearest_neighbors(
            result_vectors, treatment_profiles['profile'])
        moas = np.array(treatment_profiles['moa'].iloc[nearest_neighbors])

        target_moa = treatment_profiles[self.compound]['moa']
        print(target_moa)
        accuracy = np.mean(moas == target_moa)
        log.info('Accuracy for MOA canceling experiment: %.3f', accuracy)

        count_map = {n: 0 for n in nearest_neighbors}
        for neighbor in nearest_neighbors:
            count_map[neighbor] += 1
        counts = list(count_map.items())
        counts.sort(key=lambda p: p[1], reverse=True)
        top_k_indices, top_k_counts = list(zip(*counts[:3]))
        top_k_moas = treatment_profiles['moa'].iloc[top_k_indices]

        top_k_pairs = ((m, str(c)) for m, c in zip(top_k_moas, top_k_counts))
        top_k_string = ', '.join(' '.join(i) for i in top_k_pairs)

        log.info('Top 3 MOAs for MOA canceling experiment (correct: %s): %s',
                 target_moa, top_k_string)

        lhs = target_moa[:len(result_vectors)]
        rhs = base = ['DMSO'] * len(lhs)
        assert len(lhs) == len(rhs) == len(base) == len(moas), (len(lhs),
                                                                len(rhs),
                                                                len(base),
                                                                len(moas))
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
