import torch
from torch.utils.data import IterableDataset

from infinibatch import common


class ChunkedDataset(IterableDataset):
    def __init__(self, path, shuffle=False, buffer_size=1024, transform=None):
        super().__init__()
        self.dataset = common.chunked_dataset.ChunkedDataset(path, shuffle=shuffle, buffer_size=buffer_size, transform=transform)


    def __iter__(self):
        return self.dataset.__iter__()
