from abc import abstractmethod
import collections
import copy
import gzip
from itertools import cycle, islice
import os
from random import Random
from typing import Any, Callable, Iterable, Iterator, Generator, List, NamedTuple, Optional, Union


# TODO for first release:
# - implement new version of BufferedShuffleIterator that has smaller checkpoints
# - implement prefetching with a buffer (possibly at the end of the pipeline) to avoid latency spikes
# - modify chunked_readlines_iterator to also work on uncompressed data, or even more general data formats
#    - one possible design is to replace the hard-coded "gzip.open ... lines.split" with a lambda passed to the constructor
# - add type checks to guarantee that input iterators are checkpointable?

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
    return collections.namedtuple("namedtuple_from", members.keys())(**members)


def _advance_iterator(iterator: Iterator, n: int):
    """ Little helper to advance an iterator by n items """
    for _ in range(n):
        next(iterator)
    return n


class CheckpointableIterator(collections.abc.Iterator):
    """
    Abstract base class for iterators that are checkpointable
    
    The interface (getstate, setstate) is inspired by Python's random package.
    """
    def __iter__(self):
        return self

    @abstractmethod
    def getstate(self) -> NamedTuple:
        pass

    @abstractmethod
    def setstate(self, checkpoint: Optional[NamedTuple]):
        pass

    @abstractmethod
    def __next__(self):
        pass


class NativeCheckpointableIterator(CheckpointableIterator):
    def __init__(self, iterable: Iterable):
        """
        Simple checkpointable wrapper around native Python iterable.
        This version just replays the iterator all the way to the checkpoint, which will
        make it inefficient for some important use cases.

        Warning: This class cannot be used with Iterators (as opposed to Iterables), which have an __iter__ function that simply returns self, but does not reset.
        """
        # check whether iterable is iterable or iterator:
        # if the variable iterable contains an iterator, the function __iter__ returns self
        # if the variable iterable is an actual iterator, it should not return self
        if iter(iterable) is iterable:  
            raise ValueError('It looks like you are passing an iterator instead of an iterable. This is not supported and can cause undefined behavior when used with checkpointing.')
        self._input_iterable = iterable
        self.setstate(None)

    def getstate(self) -> NamedTuple:
        return _namedtuple_from(consumed_items=self._consumed_items)

    def setstate(self, checkpoint: Optional[NamedTuple]):
        self._iterator = iter(self._input_iterable)
        self._consumed_items = _advance_iterator(self._iterator, checkpoint.consumed_items) if checkpoint is not None else 0

    def __next__(self):
        item = next(self._iterator)  # call this before increasing _consumed_items to correctly handle the case when a StopIteration exception is thrown
        self._consumed_items += 1
        return item


class InfinitePermutationIterator(CheckpointableIterator):
    def __init__(self, items: Iterator, seed: Optional[int]=None, shuffle: bool=True, num_instances: int=1, instance_rank: int=0):
        """
        Infinitely generates permutations of the items in the given iterable.

        Unlike most classes here, this one loads all items into RAM. For example, this is used
        for randomizing the pathnames of data blocks read by chunked_readlines_iterator.

        Args:
            iterator: input iterator
            seed: random seed used for shuffling (or None)
            shuffle: set False to bypass the shuffling. Then this is just a checkpointed version of itertools.cycle(). (Default: True)
            num_instances: number of instances of this dataset. Meant for use with multi-process data loading, e.g., in distributed training.
            instance_rank: rank of this instance of the dataset. Meant for use with multi-process data loading, e.g., in distributed training.
        """
        self._original_items = list(items)  # keep a local copy, since items is an iterator
        self._shuffle = shuffle
        self._seed = seed
        self._num_instances = num_instances
        self._instance_rank = instance_rank
        self.setstate(None)

    def getstate(self) -> NamedTuple:
        return _namedtuple_from(
            random_state = self._random_state,  # state of random generator before generating the current shuffling of the sequence
            item_count   = self._item_count)    # how many items have already been iterated over in the current shuffling

    def setstate(self, checkpoint: Optional[NamedTuple]):
        # set iteration state. Do this outside the generator below in case getstate() is called before ever iterating
        self._random_state = checkpoint.random_state if checkpoint else None
        self._item_count   = checkpoint.item_count   if checkpoint else 0
        # We define the iteration itself as a generator for ease of implementation.
        # We could as well just have used an explicit state machine represented by class members.
        def _generate() -> Iterator:
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
                shuffled_items = self._original_items[:]  # note: if underlying iterator is checkpointable, use setstate(checkpoint.nested_state) on it
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

    def __next__(self):
        return next(self._generator)


class SelectManyIterator(CheckpointableIterator):
    def __init__(self, source_items: CheckpointableIterator, collection_selector: Callable[[Any], Iterable]):
        """
        Projects each element of a source sequence to a sequence and flattens the resulting sequences into one sequence.

        Args:
            source_items: iterable of paths to chunk files
        """
        self._source_items: CheckpointableIterator = source_items
        self._collection_selector: Callable[[Any], Iterable] = collection_selector
        self.setstate(None)

    def getstate(self) -> NamedTuple:
        return _namedtuple_from(
            nested_state = self._input_state,
            item_index   = self._flattened_item_index)

    def setstate(self, checkpoint: Optional[NamedTuple]):
        self._input_state = checkpoint.nested_state if checkpoint else None
        self._flattened_item_index  = checkpoint.item_index   if checkpoint else 0
        self._source_items.setstate(self._input_state)
        def _generate():
            skip_to_checkpoint = self._flattened_item_index
            # main loop over source source_items
            for source_item in self._source_items:
                data = self._collection_selector(source_item)
                self._flattened_item_index = 0
                if skip_to_checkpoint:
                    #print("Skipping to index", skip_to_checkpoint, file=sys.stderr)
                    self._flattened_item_index += _advance_iterator(data, skip_to_checkpoint)
                    skip_to_checkpoint = 0
                # main loop over lines
                for item in data:
                    self._flattened_item_index += 1
                    yield item
                self._input_state = self._source_items.getstate()
        self._iterator = _generate()

    def __next__(self):
        return next(self._iterator)


# @TODO: Can we seamlessly support UCS-2 files as well? C# can auto-detect. Does Python have such a facility?
def chunked_readlines_iterator(chunk_file_paths: CheckpointableIterator):
    """
    Reads text lines from zipped chunk files whose names are provided by an iterator.

    Args:
        chunk_file_paths: CheckpointableIterator of paths to chunk files
    """
    def readlines_from_zipped(textfile_path: str) -> Iterable[str]:
        #print("Reading chunk file", textfile_path, file=sys.stderr)
        with gzip.open(textfile_path, 'rt', encoding='utf-8') as f:
            return iter(f.read().splitlines())
    return SelectManyIterator(source_items=chunk_file_paths, collection_selector=readlines_from_zipped)


class BufferedShuffleIterator(CheckpointableIterator):
    def __init__(self, input_iterator: CheckpointableIterator, buffer_size: int, seed: int = 0):
        """
        Shuffles given iterable using a limited buffer.
        
        Args:
            input_iterator: checkpointable iterator or restartable iterable over input items to shuffle
            buffer_size: size of the buffer in number of items used for shuffling
            seed: random seed used for shuffling (or None)
        """
        self._input_iterator = input_iterator
        self._buffer = [None for _ in range(buffer_size)]  # maybe do this lazily?   --Yes, since user may set state immediately, then this is not needed here
        self._random = Random(seed)
        self.setstate(None)

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

    def __next__(self):
        return next(self._generator)


class TransformIterator(CheckpointableIterator):
    def __init__(self, input_iterator: CheckpointableIterator, transform: Callable[[str],Any]=None):
        """
        Applies given tranform to each data item
        
        Args:
            input_iterator: checkpointable iterator
            transform: function to be applied to each data item
        """
        self._input_iterator = input_iterator
        self._transform = transform

    def getstate(self) -> NamedTuple:
        return self._input_iterator.getstate()

    def setstate(self, checkpoint: Optional[NamedTuple]):
        self._input_iterator.setstate(checkpoint)

    def __next__(self):
        return self._transform(next(self._input_iterator))


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

    def __next__(self):
        return next(self._batch_iter)