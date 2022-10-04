import itertools
import numpy as np
from torch.utils.data.sampler import Sampler


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        # 有标签的索引
        self.primary_indices = primary_indices
        # 无标签的索引
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        # 随机打乱索引顺序
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    # print('shuffle labeled_idxs')
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    # print('shuffle unlabeled_idxs')
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


if __name__ == '__main__':
    labeled_idxs = list(range(12))
    unlabeled_idxs = list(range(12,60))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, 4, 2)
    for _ in range(2):
        i = 0
        for x in batch_sampler:
            i += 1
            print('%02d' % i, '\t', x)
