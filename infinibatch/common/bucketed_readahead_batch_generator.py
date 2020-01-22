from typing import Union, List, Iterator, Any, Callable
from random import Random
from itertools import islice

# based on Marian NMT's BatchGenerator
# Note: This could be implemented more elegantly as a generator function.
# However, that may no longer be true with checkpointing, so let's keep it as a class for now.
class BucketedReadaheadBatchGenerator:
    # parameters
    _key: Callable[[Any], Any]            # callback to define how data is sorted for purpose of batching
    _batch_size_fn: Callable[[Any], Any]  # callback to determine batch size for a given first item
    _read_ahead: int                     # how many items should be read ahead?

    # state
    _data_iter: Iterator[Any]   # iterator into _dataset
    _random: Random             # random generator
    _dataset_exhausted: bool    # set to True once we hit StopIteration on dataset
    _batch_iter: Iterator[Any]  # iterator into current set of batches

    def __init__(self, dataset, read_ahead: int, key: Callable[[Any], Any], batch_size_fn: Callable[[Any], Any], shuffle: bool=True, seed: int=None):
        # keep arguments
        self._key = key
        self._batch_size_fn = batch_size_fn
        self._read_ahead = read_ahead
        # initialize state
        if shuffle:
            self._random = Random()
            if seed is not None:
                self._random.seed(seed)
        self._data_iter = iter(dataset)
        self._dataset_exhausted = False
        self._rebuffer()  # get first set

    def _rebuffer(self):  # this is called whenever we need to create the next set of batches
        if self._dataset_exhausted: # dataset has flagged end
            raise StopIteration
        # prefetch the readahead buffer  --@TODO: This is really a Take(_read_ahead)
        lines = list(islice(self._data_iter, self._read_ahead))
        self._dataset_exhausted = (len(lines) < self._read_ahead)
        # sort by length, longest first
        lines.sort(key=lambda line: len(line), reverse=True)  # note: sort() is stable, so we won't undo any randomization besides the bucketing
        # group into batches
        cur_batch = None
        batch_size: int
        batches = list()
        for line in lines:
            if not cur_batch:
                batch_size = self._batch_size_fn(line)
                cur_batch = list()
            cur_batch.append(line)
            if len(cur_batch) >= batch_size:
                batches.append(cur_batch)
                cur_batch = None
        if cur_batch:
            batches.append(cur_batch)
        # shuffle the batches
        if self._random:
            self._random.shuffle(batches)
        # we serve from this list of randomized batches until it is exhausted
        self._batch_iter = iter(batches)

    def __next__(self):
        try:
            return next(self._batch_iter)
        except StopIteration:
            self._rebuffer()               # exhausted: get the next set of batches. May raise StopIteration.
            return next(self._batch_iter)  # should not fail

    def __iter__(self):
        return self
