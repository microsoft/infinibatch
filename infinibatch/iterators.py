from abc import ABC, abstractmethod
from collections import namedtuple
import copy
import gzip
from itertools import cycle, islice
import os
from random import Random
from typing import Any, Callable, Iterable, Iterator, Generator, List, NamedTuple, Optional, Union


# TODO for first release:
# - implement new version of BufferedShuffleIterator that has smaller checkpoints
# - implement prefetching with a buffer (possibly at the end of the pipeline) to avoid latency spikes
# - modify ChunkedDataIterator to also work on uncompressed data, or even more general data formats
#    - one possible design is to replace the hard-coded "gzip.open ... lines.split" with a lambda passed to the constructor

# TODO later:
# - make iterator pipeline work for streaming data


def _namedtuple_from(**members):
    """
    Creates a record of variables, which are then accessed by . syntax.
    Wraps namedtuple type creation and instantiation into a single call.

    Example:
        >>> r = namedtuple_from(x = 13, y = 42)
        >>> r.x
            13

    Args:
        members: values that the record is to contain

    Returns:
        A singleton named tuple that has all passed kw args as immutable class members.
    """
    return namedtuple("namedtuple_from", members.keys())(**members)


def _advance_iterator(iterator: Iterator[Any], n: int):
    """ Little helper to advance an iterator by n items """
    for _ in range(n):
        next(iterator)
    return n


class CheckpointableIterator(ABC):
    """ Abstract base class for iterators that are checkpointable """
    @abstractmethod
    def __next__(self):
        raise NotImplementedError()

    def __iter__(self):
        return self

    @abstractmethod
    def getstate(self) -> NamedTuple:
        raise NotImplementedError()

    @abstractmethod
    def setstate(self, checkpoint: Optional[NamedTuple]):
        raise NotImplementedError()


# @TODO: Can we have one that also takes an input iterator?
#        Then getstate() can inquire that one, and remember how often we have
#        advanced since the last call to getstate(). Upon setstate(), we'd
#        setstate() in the input iterator and then advance only the remaining few.
class NativeCheckpointableIterator(CheckpointableIterator):
    """
    Simple checkpointable wrapper around native Python iterable.
    This version just replays the iterator all the way to the checkpoint, which will
    make it inefficient for some important use cases.

    Note: It only works with true iterables that reset upon each call to iter().
    Iterators have an iter() method but don't reset themselves.
    """
    _input_iterable: Iterable[Any]
    _iterator: Iterator[Any]
    _consumed_items: int

    def __init__(self, iterable: Iterable[Any]):
        # @BUGBUG: We should check whether the input iterable is really a restartable iterable (and not an fake one such as an iterator)
        self._input_iterable = iterable
        self.setstate(None)

    def __next__(self):
        item = next(self._iterator)  # call this before increasing _consumed_items to correctly handle the case when a StopIteration exception is thrown
        self._consumed_items += 1
        return item

    def getstate(self) -> NamedTuple:
        return _namedtuple_from(consumed_items=self._consumed_items)

    def setstate(self, checkpoint: Optional[NamedTuple]):
        self._iterator = iter(self._input_iterable)
        self._consumed_items = _advance_iterator(self._iterator, checkpoint.consumed_items) if checkpoint is not None else 0


class InfinitePermutationIterator(CheckpointableIterator):
    """
    Infinitely generates permutations of the items in the given iterable.

    Unlike most classes here, this one loads all items into RAM. For example, this is used
    for randomizing the pathnames of data blocks read by _ChunkedDataIterator.

    Args:
        iterator: input iterator
        seed: random seed used for shuffling (or None)
        shuffle: set False to bypass the shuffling. Then this is just a checkpointed version of itertools.cycle(). (Default: True)
        num_instances: number of instances of this dataset. Meant for use with multi-process data loading, e.g., in distributed training.
        instance_rank: rank of this instance of the dataset. Meant for use with multi-process data loading, e.g., in distributed training.
    """
    # constructor arguments
    _original_items: List[Any]  # note: in this case, the source iterator is not checkpointable, hence we must instead keep a full copy
    _seed: Optional[int]
    _shuffle: bool
    _num_instances: int
    _instance_rank: int

    # output iterator that generates our output sequence
    _generator: Iterator[Any]

    # iteration state. This is returned when requesting the checkpoint, and restored when resetting to checkpoint.
    _random_state: Any
    _item_count: int

    def __init__(self, items: Iterator[Any], seed: Optional[int]=None, shuffle: bool=True, num_instances: int=1, instance_rank: int=0):
        # keep arguments for iter_from_checkpoint
        self._original_items = list(items)  # keep a local copy, since items is an iterator
        self._shuffle = shuffle
        self._seed = seed
        self._num_instances = num_instances
        self._instance_rank = instance_rank
        self.setstate(from_checkpoint=None)
    
    def __next__(self):
        return next(self._generator)

    def getstate(self) -> NamedTuple:
        return _namedtuple_from(
            random_state = self._random_state,  # state of random generator before generating the current shuffling of the sequence
            item_count   = self._item_count)    # how many items have already been iterated over in the current shuffling

    def setstate(self, from_checkpoint: Optional[NamedTuple]):
        # set iteration state. Do this outside the generator below in case getstate() is called before ever iterating
        self._random_state = from_checkpoint.random_state if from_checkpoint else None
        self._item_count   = from_checkpoint.item_count   if from_checkpoint else 0
        # We define the iteration itself as a generator for ease of implementation.
        # We could as well just have used an explicit state machine represented by class members.
        def _generate():
            # create and reset random generator
            random = Random(self._seed)
            if self._random_state is not None:  # restore the random generator's state
                random.setstate(self._random_state)
            skip_to_checkpoint = self._item_count  # items to skip in order to advance to checkpoint
            # main outer loop for infinite passes over items (reshuffle before each pass)
            while True:
                # (re-)shuffle all items
                self._random_state = random.getstate()  # remember random state before shuffling
                self._item_count   = 0
                shuffled_items = self._original_items[:]  # note: if underlying iterator is checkpointable, use setstate(from_checkpoint.nested_state) on it
                if self._shuffle:
                    random.shuffle(shuffled_items)
                shuffled_iterator = iter(shuffled_items)
                # skip initial items when restarting from checkpoint
                if skip_to_checkpoint:  # @TODO: find a way to abstract this more, so that we can plug it into the 'for' statement directly
                    self._item_count += _advance_iterator(shuffled_iterator, skip_to_checkpoint)
                    skip_to_checkpoint = 0  # done skipping
                # main inner loop over items
                for item in shuffled_iterator:
                    self._item_count += 1  # record how many items we have iterated over in this pass over the items
                    if (self._item_count-1) % self._num_instances == self._instance_rank:  # build-in islice facility
                        yield item
        self._generator = _generate()


# @TODO: Can we seamlessly support UCS-2 files as well? C# can auto-detect. Does Python have such a facility?
# @TODO: Support non-gzipped files as well
# @TODO: Is it wise to have the transform here? It will increase startup time [Liyang]
# @TODO: Shoult the splitlines() below also be passable as a lambda? Then we could read binary files. Maybe combine with transform?
class ChunkedDataIterator(CheckpointableIterator):
    _chunk_file_paths: Iterator[str]
    _transform: Callable[[str],Any]

    _iterator: Iterator[Any]

    _input_state: Union[Optional[Any], int]
    _line_index: int

    def __init__(self, chunk_file_paths: Iterator[str], transform: Callable[[str],Any]=None):
        """
        Reads data items (text lines) from chunk files. Optionally parses each item with a caller-supplied transform.

        Args:
            chunk_file_paths: iterable of paths to chunk files
            transform: transform to parse each text line (transform(str) -> Any)
        """
        if not isinstance(chunk_file_paths, CheckpointableIterator):
            chunk_file_paths = NativeCheckpointableIterator(chunk_file_paths)
        self._chunk_file_paths = chunk_file_paths
        self._transform = transform
        self.setstate(None)
    
    def setstate(self, checkpoint: Optional[NamedTuple]):
        self._input_state = checkpoint.nested_state if checkpoint else None
        self._line_index  = checkpoint.line_index   if checkpoint else 0
        self._chunk_file_paths.setstate(self._input_state)
        def _generate():
            skip_to_checkpoint = self._line_index
            # main loop over chunk files
            for chunk_file_path in self._chunk_file_paths:
                #print("Reading chunk file", chunk_file_path, file=sys.stderr)
                with gzip.open(chunk_file_path, 'rt', encoding='utf-8') as f:
                    data = iter(f.read().splitlines())
                self._line_index = 0
                if skip_to_checkpoint:
                    #print("Skipping to index", skip_to_checkpoint, file=sys.stderr)
                    self._line_index += _advance_iterator(data, skip_to_checkpoint)
                    skip_to_checkpoint = 0
                # main loop over lines
                for item in data:
                    self._line_index += 1
                    yield item
                self._input_state = self._chunk_file_paths.getstate()
        self._iterator = _generate()
    
    def getstate(self) -> NamedTuple:
        return _namedtuple_from(
            nested_state = self._input_state,
            line_index   = self._line_index)

    def __next__(self):
        item = next(self._iterator)
        if self._transform is not None:  # transform is applied on the fly here
            item = self._transform(item)
        return item


class BufferedShuffleIterator(CheckpointableIterator):
    _input_iterator: Iterator[Any]
    _buffer: List[Optional[Any]]
    _random: Random
    _generator: Iterator[Any]

    def __init__(self, input_iterator: Union[CheckpointableIterator,Iterable[Any]], buffer_size: int, seed: int = 0):
        """
        Shuffles given iterable using a limited buffer.
        
        Arguments:
        input_iterator -- checkpointable iterator or restartable iterable over input items to shuffle
        buffer_size -- size of the buffer in number of items used for shuffling
        seed -- random seed used for shuffling (or None)
        """
        if not isinstance(input_iterator, CheckpointableIterator):  # @TODO: factor this into a helper method
            input_iterator = NativeCheckpointableIterator(input_iterator)
        self._input_iterator = input_iterator
        self._buffer = [None for _ in range(buffer_size)]  # maybe do this lazily?   --Yes, since user may set state immediately, then this is not needed here
        self._random = Random(seed)
        self.setstate(None)

    def __next__(self):
        return next(self._generator)

    def _generate(self):
        # shuffle data with a buffer:
        # this is similar to what the Fisher-Yates shuffle does,
        # but modified to run with a constant-size buffer
        # see https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
        # this was inspired by an algorithm implemented in Kaldi
        # see https://kaldi-asr.org/doc/nnet-shuffle-egs_8cc.html
        for item in self._input_iterator:
            index = self._random.randrange(0, len(self._buffer))
            result = None
            if self._buffer[index] is not None:
                result = self._buffer[index]
            self._buffer[index] = item
            # only yield value once buffer is updated to allow for correct checkpointing!
            if result is not None:
                yield result

        # flush buffer
        while self._buffer:
            item = self._buffer.pop()
            if item is not None:
                yield item

    def getstate(self) -> NamedTuple:
        return _namedtuple_from(
            nested_checkpoint = self._input_iterator.getstate(),
            buffer            = copy.deepcopy(self._buffer),
            random_state      = self._random.getstate())

    def setstate(self, checkpoint: Optional[NamedTuple]):
        if checkpoint:
            self._input_iterator.setstate(checkpoint.nested_checkpoint)
            self._buffer = checkpoint.buffer
            self._random.setstate(checkpoint.random_state)
            # @TODO: Can we add a comment how the flush part is handled?
        else:
            self._input_iterator.setstate(None)
        self._generator = self._generate()


# @TODO: Support non-zipped files.
# @TODO: Support index files?
# @TODO: This is no more than a factory function. We should change it to one
class ChunkedDatasetIterator(CheckpointableIterator):
    _iterator: Iterator[Any]  # our output iterator

    def __init__(self, paths: Union[str, Iterable[str]], shuffle: bool=True, buffer_size: int=2**20, transform:Callable[[Any],Any]=None, seed: Optional[int]=None, num_instances: int=1, instance_rank: int=0):
        """
        Dataset reading data from gzipped chunks.

        This dataset infinitely repeats the data.

        Arguments:
        paths -- path, or list of paths, of directory containing dataset, i.e., a collection of .gz-files containing compressed text
        shuffle -- if true, the data is shuffled
        buffer_size -- size of the buffer in number of samples / data items used for shuffling
        transform -- transform to be applied to each data item (transform(Any) -> Any)
        seed -- random seed (or None)
        num_instances -- number of instances of this dataset. Meant for use with multi-process data loading, e.g., in distributed training.
        instance_rank -- rank of this instance of the dataset. Meant for use with multi-process data loading, e.g., in distributed training.
        """
        if isinstance(paths, str):  # handle single string
            paths = [paths]
        # set up the chunk reader
        chunk_file_paths = [  # enumerate all .gz files in the given paths
            os.path.join(path, subpath.name)
            for path in paths
            for subpath in os.scandir(path)
            if subpath.is_file() and subpath.name.endswith('.gz')
        ]
        chunk_file_paths.sort()  # make sure file order is always the same, independent of OS
        chunks  = InfinitePermutationIterator(chunk_file_paths, seed, shuffle=shuffle, num_instances=num_instances, instance_rank=instance_rank)
        # set up the item reader
        samples = ChunkedDataIterator(chunks, transform)
        # set up the item randomizer
        if shuffle:
            # use different seed for BufferedShuffleGenerator
            buffered_shuffle_iterator_seed = seed
            if buffered_shuffle_iterator_seed is not None:
                buffered_shuffle_iterator_seed += 1
            samples = BufferedShuffleIterator(samples, buffer_size, buffered_shuffle_iterator_seed)
        
        # this is what we are serving out
        self._iterator = samples
        self.setstate(None)
    
    def setstate(self, checkpoint):
        self._iterator.setstate(checkpoint)
    
    def getstate(self):
        return self._iterator.getstate()
    
    def __next__(self):
        return next(self._iterator)


# However, that may no longer be true with checkpointing, so let's keep it as a class for now.
class BucketedReadaheadBatchDatasetIterator(CheckpointableIterator):
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
        self.setstate(None)
    
    def getstate(self):
        return _namedtuple_from(
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

    def setstate(self, checkpoint: Optional[NamedTuple]):
        self._input_state  = checkpoint.input_state  if checkpoint else None
        self._random_state = checkpoint.random_state if checkpoint else None
        self._num_served   = checkpoint.num_served   if checkpoint else 0
        # checkpointing: restore to start of current set of batches
        self._data_iter.setstate(self._input_state)
        if self._random_state:
            self._random.setstate(self._random_state)
        self._dataset_exhausted = False
        def _generate():
            skip_to_checkpoint = self._num_served
            dataset_exhausted = False
            while not dataset_exhausted:
                # prefetch the readahead buffer
                self._input_state = self._data_iter.getstate()
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
