import copy
import numpy as np
from torch.utils.data.sampler import Sampler


class BalancedSampler(Sampler):
    """
    Args:
        labels (ndarray): an array of labels.
        num_samples_per_class (int): number of samples to draw per class.
    """

    def __init__(self, labels, num_samples_per_class):
        class_counts = np.bincount(labels)
        indices = np.arange(len(labels))
        self.num_samples = len(class_counts) * num_samples_per_class
        self.labels = labels
        self.num_samples_per_class = num_samples_per_class
        self.class_counts = class_counts
        self.indices = indices

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        sampled_indices = []
        for class_, count in enumerate(self.class_counts):
            if count >= self.num_samples_per_class:
                replace = False
            elif count < self.num_samples_per_class:
                replace = True
            sampled_indices.extend(
                np.random.choice(
                    self.indices[self.labels == class_],
                    size=self.num_samples_per_class,
                    replace=replace,
                ).tolist()
            )
        sampled_indices = np.random.permutation(sampled_indices).tolist()
        return iter(sampled_indices)


class SemiSupervisedSampler(Sampler):
    def __init__(self, cfg, supervised_sampler, semi_supervised_dataset):
        self.supervised_sampler = supervised_sampler
        self.semi_supervised_dataset = semi_supervised_dataset
        self.num_supervised_dataset = len(supervised_sampler.labels)

    def __iter__(self):
        supervised_idxs = list(iter(self.supervised_sampler))

        semi_supervised_idxs = np.arange(len(self.semi_supervised_dataset))
        # offset by length of supervised dataset
        semi_supervised_idxs += self.num_supervised_dataset

        idxs = np.append(semi_supervised_idxs, supervised_idxs)
        idxs = np.random.permutation(idxs).tolist()
        return iter(idxs)

    def __len__(self):
        return len(self.supervised_sampler) + len(self.semi_supervised_dataset)


class RandomClassSampler(Sampler):
    """
    Randomly sample N classes and then randomly sample k instances
    from each class.
    """

    def __init__(self, labels, num_samples_per_class, batch_size, max_iters):
        self.indices = np.arange(len(labels))
        self.labels = labels
        self.classes = np.unique(self.labels)[:100]
        self.k = num_samples_per_class
        self.batch_size = batch_size
        self.num_classes_per_batch = self.batch_size // self.k
        self.max_iters = max_iters

    def __len__(self):
        return self.max_iters

    # def __repr__(self):

    def _make_batch(self):
        batch_idxs = {}
        from tqdm import tqdm

        for class_ in tqdm(self.classes):
            idxs = self.indices[self.labels == class_]
            if len(idxs) < self.k:
                idxs = np.append(
                    idxs, np.random.choice(idxs, size=self.k - len(idxs), replace=True)
                )
            idxs = np.random.permutation(idxs)
            batch_idxs[class_] = [
                idxs[i * self.k : (i + 1) * self.k].tolist()
                for i in range(len(idxs) // self.k)
            ]

        avail_classes = copy.deepcopy(self.classes).tolist()
        return batch_idxs, avail_classes

    def __iter__(self):
        batch_idxs, avail_classes = self._make_batch()
        batch = []
        for _ in range(self.max_iters):
            if len(avail_classes) < self.num_classes_per_batch:
                batch_idxs, avail_classes = self._make_batch()
            selected_classes = np.random.choice(
                avail_classes, size=self.num_classes_per_batch, replace=False
            )
            for class_ in selected_classes:
                batch.extend(batch_idxs[class_].pop(0))
                if len(batch_idxs[class_]) == 0:
                    avail_classes.remove(class_)
        return iter(batch)
