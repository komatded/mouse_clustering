import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


class TripletGenerator:

    def __init__(self, pad_size: int, positives_per_anchor: int, negatives_per_anchor: int, drop_time_line: bool,
                 random_state=42):
        self.pad_size = pad_size
        self.positives_per_anchor = positives_per_anchor
        self.negatives_per_anchor = negatives_per_anchor
        self.drop_time_line = drop_time_line
        self.random_state = random_state

    def create_data_generator(self, data: pd.DataFrame, batch_size: int):
        n_batches = len(data.cookie.unique()) * self.positives_per_anchor * \
                    (self.positives_per_anchor - 1) * self.negatives_per_anchor // batch_size + 1

        def generator():
            anchors, positives, negatives = list(), list(), list()
            while True:
                for anchor_cookie in data.cookie.unique():
                    positive_data = data[data.cookie == anchor_cookie]
                    positive_data = positive_data.sample(n=min(self.positives_per_anchor, len(positive_data)),
                                                         random_state=self.random_state)
                    positive_data = positive_data.mouse_track.values
                    negative_data = data[data.cookie != anchor_cookie]
                    negative_data = negative_data.sample(n=min(self.negatives_per_anchor, len(negative_data)),
                                                         random_state=self.random_state)
                    negative_data = negative_data.mouse_track.values
                    for anchor, positive, negative in self._generate_triplets(positive_data, negative_data):
                        anchors.append(anchor)
                        positives.append(positive)
                        negatives.append(negative)
                        if len(anchors) == batch_size:
                            yield ([np.array(anchors)[:, :, 1 * self.drop_time_line:],
                                    np.array(positives)[:, :, 1 * self.drop_time_line:],
                                    np.array(negatives)[:, :, 1 * self.drop_time_line:]],
                                   np.ones(len(anchors)))
                            anchors, positives, negatives = list(), list(), list()
        return generator(), n_batches

    def _generate_triplets(self, positives, negatives):
        if len(positives) == 1:
            for negative in negatives:
                anchor, positive, negative = self._pad(positives[0]), self._pad(positives[0]), self._pad(negative)
                yield anchor, positive, negative
        for anchor, positive in itertools.combinations(positives, 2):
            anchor, positive = self._pad(anchor), self._pad(positive)
            for negative in negatives:
                negative = self._pad(negative)
                yield anchor, positive, negative

    def _pad(self, array):
        output = list(array)[:self.pad_size] + [[0, 0, 0]] * (self.pad_size - len(array))
        return output


class TripletGeneratorBase:

    def __init__(self, positives_per_anchor: int, negatives_per_anchor: int, random_state=42):
        self.positives_per_anchor = positives_per_anchor
        self.negatives_per_anchor = negatives_per_anchor
        self.random_state = random_state

    def create_data_generator(self, data: pd.DataFrame, batch_size: int):
        n_batches = len(data.cookie.unique()) * self.positives_per_anchor * \
                    (self.positives_per_anchor - 1) * self.negatives_per_anchor // batch_size + 1

        def generator():
            anchors, positives, negatives = list(), list(), list()
            while True:
                for anchor_cookie in data.cookie.unique():
                    positive_data = data[data.cookie == anchor_cookie]
                    positive_data = positive_data.sample(n=min(self.positives_per_anchor, len(positive_data)),
                                                         random_state=self.random_state)
                    positive_data = positive_data.drop('cookie', axis=1).to_numpy()
                    negative_data = data[data.cookie != anchor_cookie]
                    negative_data = negative_data.sample(n=min(self.negatives_per_anchor, len(negative_data)),
                                                         random_state=self.random_state)
                    negative_data = negative_data.drop('cookie', axis=1).to_numpy()
                    for anchor, positive, negative in self._generate_triplets(positive_data, negative_data):
                        anchors.append(anchor)
                        positives.append(positive)
                        negatives.append(negative)
                        if len(anchors) == batch_size:
                            anchors_norm, positives_norm, negatives_norm = normalize(anchors), normalize(positives), normalize(negatives)
                            yield ([np.array(anchors_norm),
                                    np.array(positives_norm),
                                    np.array(negatives_norm)],
                                   np.ones(len(anchors)))
                            anchors, positives, negatives = list(), list(), list()
        return generator(), n_batches

    @staticmethod
    def _generate_triplets(positives, negatives):
        for anchor, positive in itertools.combinations(positives, 2):
            for negative in negatives:
                yield anchor, positive, negative
