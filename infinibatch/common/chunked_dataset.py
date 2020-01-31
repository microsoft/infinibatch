import gzip
import itertools
import os
from random import Random
from typing import Union, Iterable, Iterator, List, Any, Callable, Optional

# Note: All Iterables here are now Iterators

# wrapper around standard Python Iterators
# This class itself has no state. Instead, state is encapsulated by three lambdas that are passed in.
# Implements iterator protocol, i.e. next() and iter(), but also get_checkpoint() and iter_from_checkpoint().
class CheckpointedIteratorWrapper():
    _next: Callable[[], Any]
    get_checkpoint:       Callable[[], List[Any]]       # these are callable member functions
    iter_from_checkpoint: Callable[[List[Any]], None]

    def __init__(self, next: Callable[[], Any], get_checkpoint: Callable[[], List[Any]], iter_from_checkpoint: Callable[[List[Any]], None]):
        self.get_checkpoint       = get_checkpoint        # these are now callable members
        self.iter_from_checkpoint = iter_from_checkpoint
        self._next                = next                  # does not work with __next__, which is special   
    
    def __next__(self):
        return self._next()
    
    def __iter__(self):
        return self


# simple class definition and instance creation in one
# e.g. x = Struct(a=13, b=42) then use x.a and x.b
class Struct:
    def __init__(self, **args_dict):
        self.__dict__.update(args_dict)
    def copy(self):
        return Struct(**dict(self.__dict__.items()))


def infinite_permutation_iterator(items: Iterator[Any], seed: Optional[int], from_checkpoint: Optional[List[Any]] = None):
    """
    Infinitely generates permutations of the items in the given iterable.

    Unlike most classes here, this one loads all items into RAM. For example, this is used
    for randomizing the pathnanes of data blocks read by _IterableChunkedData.

    Arguments:
    iterator -- input iterator
    seed -- random seed used for shuffling (or None)
    from_checkpoint -- checkpoint info obtained by get_checkpoint(), will advance to this point
    """
    original_items = list(items)  # keep a local copy, since items is an iterator

    current_checkpoint_state = Struct(random_state = None, item_count = 0)  # current checkpoint state

    def generator():
        # create and reset random generator
        random = Random(seed)
        if from_checkpoint is not None and from_checkpoint[0].random_state is not None:  # restore the shuffled_items array
            random.setstate(from_checkpoint[0].random_state)
        skip_to_checkpoint = from_checkpoint[0].item_count if from_checkpoint is not None else 0
        # loop over items, reshuffle before each pass
        while True:
            # (re-)shuffle all items
            current_checkpoint_state.random_state = random.getstate()  # remember random state before shuffling
            current_checkpoint_state.item_count   = 0
            shuffled_items = original_items[:]
            random.shuffle(shuffled_items)
            shuffled_iterator = iter(shuffled_items)
            # skip initial items when restarting from checkpoint
            if skip_to_checkpoint:
                for i in range(skip_to_checkpoint):
                    next(shuffled_iterator)
                current_checkpoint_state.item_count += skip_to_checkpoint
                skip_to_checkpoint = 0  # only the first time
            # loop over items
            for item in shuffled_iterator:
                current_checkpoint_state.item_count += 1  # record how many items we have served from this pass over the items
                yield item

    iterator = generator()
    return CheckpointedIteratorWrapper( # wrap in a class that implements both iterator and checkpointing protocols
            next                 = lambda: next(iterator),
            get_checkpoint       = lambda: [current_checkpoint_state.copy()], # uses a list so we can embed nested checkpoint states 
            iter_from_checkpoint = lambda from_checkpoint: infinite_permutation_iterator(original_items, seed, from_checkpoint))


class InfinitePermutationIterator:  # ...how to say in Python: "implements interfaces Iterator[Any], Checkpointing"
    """
    Infinitely generates permutations of the items in the given iterable.

    Unlike most classes here, this one loads all items into RAM. For example, this is used
    for randomizing the pathnanes of data blocks read by _IterableChunkedData.

    Arguments:
    iterator -- input iterator
    seed -- random seed used for shuffling (or None)
    from_checkpoint -- checkpoint info obtained by get_checkpoint(), will advance to this point
    """
    # constructor arguments
    _original_items: List[Any]  # if source iterator is not checkpointable, we must instead keep a full copy
    _seed: Optional[int]

    # output iterator that generates our output sequence
    _iterator: Iterator[Any]

    # iteration state. This is returned when requesting the checkpoint, and restored when resetting to checkpoint.
    _random_state: Any
    _item_count: int

    def __init__(self, items: Iterator[Any], seed: Optional[int], from_checkpoint: Optional[List[Any]] = None):
        # keep arguments for iter_from_checkpoint
        self._original_items = list(items)  # keep a local copy, since items is an iterator
        self._seed = seed
        self.__setstate__(from_checkpoint)

    # Iterator protocol:
    def __iter__(self):
        return iter(self._iterator)  # or better return self, so that returned iterator is still checkpointed?
    
    def __next__(self):
        return next(self._iterator)

    # Checkpointing protocol:
    # Names are inspired by pickle https://docs.python.org/2/library/pickle.html.
    # But they look ugly when called from user code. We can use the non-__ names, like Random.
    def __getstate__(self):
        return [Struct(random_state = self._random_state, item_count = self._item_count)]

    def __setstate__(self, from_checkpoint):
        # We define the iteration itself as a generator for ease of implementation.
        # We could as well just have used an explicit state machine represented by class members.
        def _generator():
            # create and reset random generator
            random = Random(self._seed)
            if from_checkpoint is not None and from_checkpoint[0].random_state is not None:  # restore the shuffled_items array
                random.setstate(from_checkpoint[0].random_state)
            skip_to_checkpoint = from_checkpoint[0].item_count if from_checkpoint is not None else 0
            # main outer loop for infinite passes over items (reshuffle before each pass)
            while True:
                # (re-)shuffle all items
                self._random_state = random.getstate()  # remember random state before shuffling
                self._item_count   = 0
                shuffled_items = self._original_items[:]
                random.shuffle(shuffled_items)
                shuffled_iterator = iter(shuffled_items)
                # skip initial items when restarting from checkpoint
                if skip_to_checkpoint:
                    for i in range(skip_to_checkpoint):
                        next(shuffled_iterator)
                    self._item_count += skip_to_checkpoint
                    skip_to_checkpoint = 0  # only the first time
                # main inner loop over items
                for item in shuffled_iterator:
                    self._item_count += 1  # record how many items we have served from this pass over the items
                    yield item
        self._iterator = _generator()

    # alternative checkpointing protocol:
    # To reset to checkpoint, a new instance of this class is created (instead of just the inner _iterator).
    def get_checkpoint(self):  # this is identical to getstate()
        return [Struct(random_state = self._random_state, item_count = self._item_count)]
    
    def iter_from_checkpoint(self, from_checkpoint):
        return InfinitePermutationIterator(self._original_items, self._seed, from_checkpoint)


class _IterableInfinitePermutation:
    _iterable: Iterable[Any]
    _seed: Optional[int]

    def __init__(self, iterable: Iterable[Any], seed: Optional[int]):
        """
        Infinitely generates permutations of the items in the given iterable.

        Unlike most classes here, this one loads all items into RAM. For example, this is used
        for randomizing the pathnmaes of data blocks read by _IterableChunkedData.

        Arguments:
        iterable -- input iterable
        seed -- random seed used for shuffling (or None)
        """
        self._iterable = iterable
        self._seed = seed

    def __iter__(self):
        random = Random(self._seed)
        items = list(self._iterable)
        while True:
            random.shuffle(items)
            for item in items:
                yield item


# @TODO: Can we seamlessly support UCS-2 files as well? C# can auto-detect. Does Python have such a facility?
# @TODO: Support non-gzipped files as well
class _IterableChunkedData:
    _chunk_file_paths: Iterable[str]
    def __init__(self, chunk_file_paths: Iterable[str]):
        """
        Reads data from chunks.

        Arguments:
        chunk_file_paths -- iterable of paths to chunk files
        """
        self._chunk_file_paths = chunk_file_paths

    def __iter__(self):
        for chunk_file_path in self._chunk_file_paths:
            with gzip.open(chunk_file_path, 'rt', encoding='utf-8') as f:
                data = f.read().splitlines()
            for item in data:
                yield item


class _IterableBufferedShuffler:
    _iterable: Iterable[Any]
    _buffer_size: int
    _seed: Optional[int]

    def __init__(self, iterable: Iterable[Any], buffer_size: int, seed: Optional[int]):
        """
        Shuffles given iterable using a buffer.
        
        Arguments:
        iterable -- input iterable over items to shuffle
        buffer_size -- size of the buffer in number of items used for shuffling
        seed -- random seed used for shuffling (or None)
        """
        self._iterable = iterable
        self._buffer_size = buffer_size
        self._seed = seed

    def __iter__(self):
        # shuffle data with a buffer:
        # this is similar to what the Fisher-Yates shuffle does,
        # but modified to run with a constant-size buffer
        # see https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
        # this was inspired by an algorithm implemented in Kaldi
        # see https://kaldi-asr.org/doc/nnet-shuffle-egs_8cc.html
        random = Random(self._seed)
        buffer = [None for _ in range(self._buffer_size)]
        for item in self._iterable:
            index = random.randrange(0, len(buffer))
            if buffer[index] is not None:
                yield buffer[index]
            buffer[index] = item

        # flush buffer
        for item in buffer:
            if item is not None:
                yield item


# @TODO: Support non-zipped files.
# @TODO: Change default buffer size to a more reasonable value.
# @TODO: Support index files?
class IterableChunkedDataset:
    _chunk_file_paths: Union[str, Iterable[str]]
    _shuffle: bool
    _buffer_size: int
    _transform: Callable[[Any], Any] # @TODO: specify the signature
    _seed: Optional[int]
    _num_instances: int
    _instance_rank: int

    def __init__(self, paths: Union[str, Iterable[str]], shuffle: bool=True, buffer_size: int=2**20, transform=None, seed: Optional[int]=None, num_instances: int=1, instance_rank: int=0):
        """
        Dataset reading data from gzipped chunks.

        This dataset infinitely repeats the data.

        Arguments:
        paths -- path, or list of paths, of directory containing dataset, i.e., a collection of .gz-files containing compressed text
        shuffle -- if true, the data is shuffled
        buffer_size -- size of the buffer in number of samples / data items used for shuffling
        transform -- transform to be applied to each data item  --@TODO: specify its signature
        seed -- random seed (or None)
        num_instances -- number of instances of this dataset. Meant for use with multi-process data loading, e.g., in distributed training.
        instance_rank -- rank of this instance of the dataset. Meant for use with multi-process data loading, e.g., in distributed training.
        """
        if isinstance(paths, str):  # handle single string
            paths = [paths]
        self._chunk_file_paths = []
        for path in paths:
            for subpath in os.scandir(path):
                if subpath.is_file() and subpath.name.endswith('.gz'):
                    self._chunk_file_paths.append(os.path.join(path, subpath.name))
        self._chunk_file_paths.sort()  # make sure file order is always the same, independent of OS
        self._shuffle = shuffle
        self._buffer_size = buffer_size
        self._transform = transform
        self._seed = seed
        self._num_instances = num_instances
        self._instance_rank = instance_rank

    def __iter__(self):
        if not self._shuffle:
            chunks = itertools.cycle(self._chunk_file_paths)
        else:
            chunks = infinite_permutation_iterator(self._chunk_file_paths, self._seed)
        if self._num_instances > 1:
            chunks = itertools.islice(chunks, self._instance_rank, None, self._num_instances)
        
        samples = _IterableChunkedData(chunks)
        if self._shuffle:
            # use different seed for BufferedShuffleGenerator
            buffered_shuffle_iterator_seed = self._seed
            if buffered_shuffle_iterator_seed is not None:
                buffered_shuffle_iterator_seed += 1
            samples = _IterableBufferedShuffler(samples, self._buffer_size, buffered_shuffle_iterator_seed)
        if self._transform is not None:
            samples = (self._transform(item) for item in samples)
        return iter(samples)

# TODO: make this a real test
random = Random()
for i in range(20):
    # random sequence lengths to for testing different configurations
    test_source_length        = random.randrange(5,25)
    test_first_output_length  = random.randrange(5,25)
    test_second_output_length = random.randrange(5,25)
    # source
    test_source = range(test_source_length)
    reader: Iterable[Any] = InfinitePermutationIterator(test_source, seed=i)
    # fetch a first sequence
    items0 = list(itertools.islice(reader, test_first_output_length))
    print('items0', items0)
    # fetch a second sequence
    checkpoint = reader.__getstate__()
    items1a = list(itertools.islice(reader, test_second_output_length))
    print('items1a', items1a)
    # fetch that second sequence again via checkpointing
    reader.__setstate__(checkpoint)
    items1b = list(itertools.islice(reader, test_second_output_length))
    print('items1b', items1b)
    # must be the same
    assert items1a == items1b
