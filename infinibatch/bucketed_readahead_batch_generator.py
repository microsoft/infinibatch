from typing import Union, List, Iterator, Any, Callable, Optional, NamedTuple
from random import Random
from itertools import islice
from .iterators import ICheckpointableIterator, namedtuple_from, _advance_iterator


# Note: This could be implemented more elegantly as a generator function.
# However, that may no longer be true with checkpointing, so let's keep it as a class for now.
class BucketedReadaheadBatchDatasetIterator(ICheckpointableIterator):
    """
    Iterates over items from a Dataset and group items of similar length into batches.

    The algorithm reads a head a certain number of lines (e.g. 10 million), sorts them by
    length, and them groups them into batches from start to end. The sort is stable, such
    that prior randomization is not undone (except for the length grouping). The batch size
    is dynamic, and determined by a user-provided callback.

    This is based on Marian NMT's BatchGenerator.

    Args:
        dataset: The data set that is read from. Typically this is an infinite source.
        read_ahead: Number of items to fetch ahead for grouping purposes.
        key: User-provided callback to define how data is sorted for purpose of batching.
        batch_size: Batch size in number of items. Either an integer or a callback to determine batch size for a given first batch item.
        shuffle: Pass False to not randomize the batches. (default: True)
        seed: Random seed for batch shuffling.
    """
    # parameters
    _key: Callable[[Any], Any]
    _batch_size: Union[int,Callable[[Any], int]]
    _read_ahead: int

    # state
    _data_iter: Iterator[Any]   # iterator into _dataset
    _random: Random             # random generator
    _dataset_exhausted: bool    # set to True once we hit StopIteration on dataset
    _batch_iter: Iterator[Any]  # iterator into current set of batches
    _input_state: NamedTuple    # state of input before reading the current set of batches
    _num_served: int            # number of batches served from the current set of batches

    def __init__(self, dataset, read_ahead: int, key: Callable[[Any], Any], batch_size: Union[int,Callable[[Any], int]], shuffle: bool=True, seed: int=None):
        # keep arguments
        self._key = key
        self._batch_size = batch_size
        self._read_ahead = read_ahead
        # initialize state
        if shuffle:
            self._random = Random()
            if seed is not None:
                self._random.seed(seed)
        self._data_iter = iter(dataset)
        self.__setstate__(None)
    
    def __getstate__(self):
        return namedtuple_from(
            input_state  = self._input_state,
            random_state = self._random_state,
            num_served   = self._num_served)
    
    def _create_batches(self, items: List[Any]) -> List[List[Any]]:  # helper to form batches from a list of items
            # sort by length, longest first
            items.sort(key=self._key, reverse=True)  # note: sort() is stable, so we won't undo any randomization besides the bucketing
            # group into batches
            cur_batch = None
            batches = []
            for item in items:
                if not cur_batch:
                    batch_size: int = self._batch_size if isinstance(self._batch_size, int) else \
                                      self._batch_size(item)
                    cur_batch = []
                cur_batch.append(item)
                if len(cur_batch) >= batch_size:  # this batch is full
                    batches.append(cur_batch)
                    cur_batch = None
            if cur_batch:
                batches.append(cur_batch)
            return batches

    def __setstate__(self, checkpoint: Optional[NamedTuple]):
        self._input_state  = checkpoint.input_state  if checkpoint else None
        self._random_state = checkpoint.random_state if checkpoint else None
        self._num_served   = checkpoint.num_served   if checkpoint else 0
        # checkpointing: restore to start of current set of batches
        self._data_iter.__setstate__(self._input_state)
        if self._random_state:
            self._random.setstate(self._random_state)
        self._dataset_exhausted = False
        def _generate():
            skip_to_checkpoint = self._num_served
            dataset_exhausted = False
            while not dataset_exhausted:
                # prefetch the readahead buffer
                self._input_state = self._data_iter.__getstate__()
                self._random_state = self._random.getstate() if self._random else None
                items = list(islice(self._data_iter, self._read_ahead))
                dataset_exhausted = (len(items) < self._read_ahead)
                # create batches
                batches = self._create_batches(items)
                # shuffle the batches
                if self._random:
                    self._random.shuffle(batches)
                # on first loop iteration, restore iterator inside batches from checkpoint
                batches = iter(batches)
                self._num_served = _advance_iterator(batches, skip_to_checkpoint)
                skip_to_checkpoint = 0
                # main loop over batches in current read-ahead section
                for batch in batches:
                    self._num_served += 1
                    yield batch
        self._batch_iter = _generate()

    def __next__(self):
        return next(self._batch_iter)
