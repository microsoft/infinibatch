from typing import Union, Iterable, Any

import torch
from torch.utils.data import IterableDataset

from infinibatch import common


class ChunkedDataset(IterableDataset):
    def __init__(self, paths: Union[str, Iterable[str]], shuffle: bool=True, buffer_size: int=2**20, transform=None, seed: int=None, world_size: int=1, rank: int=0, num_workers_per_rank: int=1):
        super().__init__()
        self.rank = rank
        self.num_workers_per_rank = num_workers_per_rank
        # instance_rank is set assuming that num_workers_per_rank = 1 and adapted dynamically in __iter__
        self.dataset = common.chunked_dataset.ChunkedDataset(paths, shuffle=shuffle, buffer_size=buffer_size, transform=transform, seed=seed, num_instances=world_size*num_workers_per_rank, instance_rank=rank)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            self.dataset._instance_rank = self.rank
        else:
            assert worker_info.num_workers == self.num_workers_per_rank
            self.dataset._instance_rank = self.rank * self.num_workers_per_rank + worker_info.id
        return self.dataset.__iter__()
