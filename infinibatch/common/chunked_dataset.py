import gzip
from itertools import islice, cycle
import os, sys
from random import Random
from typing import Union, Iterable, Iterator, List, Any, Callable, Optional, Generator, NamedTuple
import copy


def namedtuple_from(**members):
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
    from collections import namedtuple
    return namedtuple("namedtuple_from", members.keys())(**members)


def _advance_iterator(iterator: Iterator[Any], n: int):
    """Little helper to advance an iterator by n items"""
    for _ in range(n):
        next(iterator)
    return n


class ICheckpointableIterator:  # @TODO: Can rename away the I- once done. This makes it easier during development
    def __next__(self):
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __getstate__(self) -> NamedTuple:
        raise NotImplementedError()

    def __setstate__(self, checkpoint: Optional[NamedTuple]):
        raise NotImplementedError()


class _InfinitePermutationIterator(ICheckpointableIterator):  # ...how to say in Python: "implements interfaces Iterator[Any], Checkpointing"
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
        self.__setstate__(from_checkpoint=None)

    # implementation of Iterator protocol:
    # This could go into a shared baseclass, although it seems simple enough.
    def __iter__(self):
        return self
    
    def __next__(self):
        return next(self._generator)

    # implementation of Checkpointing protocol:
    # Names are inspired by pickle https://docs.python.org/2/library/pickle.html.
    # But they look ugly when called from user code. We can use the non-__ names, like Random.
    def __getstate__(self) -> NamedTuple:
        return namedtuple_from(
            random_state = self._random_state,  # state of random generator before generating the current shuffling of the sequence
            item_count   = self._item_count)    # how many items have already been iterated over in the current shuffling

    def __setstate__(self, from_checkpoint: Optional[NamedTuple]):
        # set iteration state. Do this outside the generator below in case __getstate__() is called before ever iterating
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
                shuffled_items = self._original_items[:]  # note: if underlying iterator is checkpointable, use __setstate__(from_checkpoint.nested_state) on it
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
class _ChunkedDataIterator(ICheckpointableIterator):
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
        if not isinstance(chunk_file_paths, ICheckpointableIterator):
            chunk_file_paths = NativeCheckpointableIterator(chunk_file_paths)
        self._chunk_file_paths = chunk_file_paths
        self._transform = transform
        self.__setstate__(None)
    
    def __setstate__(self, checkpoint: Optional[NamedTuple]):
        self._input_state = checkpoint.nested_state if checkpoint else None
        self._line_index  = checkpoint.line_index   if checkpoint else 0
        self._chunk_file_paths.__setstate__(self._input_state)
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
                self._input_state = self._chunk_file_paths.__getstate__()
        self._iterator = _generate()
    
    def __getstate__(self) -> NamedTuple:
        return namedtuple_from(
            nested_state = self._input_state,
            line_index   = self._line_index)

    def __next__(self):
        item = next(self._iterator)
        if self._transform is not None:  # transform is applied on the fly here
            item = self._transform(item)
        return item


# @TODO: Can we have one that also takes an input iterator?
#        Then getstate() can inquire that one, and remember how often we have
#        advanced since the last call to getstate(). Upon setstate(), we'd
#        setstate() in the input iterator and then advance only the remaining few.
class NativeCheckpointableIterator(ICheckpointableIterator):
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
        self.__setstate__(None)

    def __next__(self):
        item = next(self._iterator)
        self._consumed_items += 1
        return item

    def __getstate__(self) -> NamedTuple:
        return namedtuple_from(
            consumed_items = self._consumed_items)

    def __setstate__(self, checkpoint: Optional[NamedTuple]):
        self._iterator = iter(self._input_iterable)
        self._consumed_items = _advance_iterator(self._iterator, checkpoint.consumed_items) if checkpoint else 0


class _BufferedShuffleIterator(ICheckpointableIterator):
    _input_iterator: Iterator[Any]
    _buffer: List[Optional[Any]]
    _random: Random
    _generator: Iterator[Any]

    def __init__(self, input_iterator: Union[ICheckpointableIterator,Iterable[Any]], buffer_size: int, seed: int = 0):
        """
        Shuffles given iterable using a limited buffer.
        
        Arguments:
        input_iterator -- checkpointable iterator or restartable iterable over input items to shuffle
        buffer_size -- size of the buffer in number of items used for shuffling
        seed -- random seed used for shuffling (or None)
        """
        if not isinstance(input_iterator, ICheckpointableIterator):  # @TODO: factor this into a helper method
            input_iterator = NativeCheckpointableIterator(input_iterator)
        self._input_iterator = input_iterator
        self._buffer = [None for _ in range(buffer_size)]  # maybe do this lazily?   --Yes, since user may set state immediately, then this is not needed here
        self._random = Random(seed)
        self.__setstate__(None)

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

    def __getstate__(self) -> NamedTuple:
        return namedtuple_from(
            nested_checkpoint = self._input_iterator.__getstate__(),
            buffer            = copy.deepcopy(self._buffer),
            random_state      = self._random.getstate())

    def __setstate__(self, checkpoint: Optional[NamedTuple]):
        if checkpoint:
            self._input_iterator.__setstate__(checkpoint.nested_checkpoint)
            self._buffer = checkpoint.buffer
            self._random.setstate(checkpoint.random_state)
            # @TODO: Can we add a comment how the flush part is handled?
        else:
            self._input_iterator.__setstate__(None)
        self._generator = self._generate()


# @TODO: Support non-zipped files.
# @TODO: Support index files?
# @TODO: This is no more than a factory function. We should change it to one
class ChunkedDatasetIterator(ICheckpointableIterator):
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
        chunks  = _InfinitePermutationIterator(chunk_file_paths, seed, shuffle=shuffle, num_instances=num_instances, instance_rank=instance_rank)
        # set up the item reader
        samples = _ChunkedDataIterator(chunks, transform)
        # set up the item randomizer
        if shuffle:
            # use different seed for BufferedShuffleGenerator
            buffered_shuffle_iterator_seed = seed
            if buffered_shuffle_iterator_seed is not None:
                buffered_shuffle_iterator_seed += 1
            samples = _BufferedShuffleIterator(samples, buffer_size, buffered_shuffle_iterator_seed)
        
        # this is what we are serving out
        self._iterator = samples
        self.__setstate__(None)
    
    def __setstate__(self, checkpoint):
        self._iterator.__setstate__(checkpoint)
    
    def __getstate__(self):
        return self._iterator.__getstate__()
    
    def __next__(self):
        return next(self._iterator)


if __name__ == '__main__':
    data_size = 10**5

    data = NativeCheckpointableIterator(iter(range(data_size)))
    shuffled_data = _BufferedShuffleIterator(data, 100)
    not_checkpointed = list(shuffled_data)

    data = NativeCheckpointableIterator(iter(range(data_size)))
    shuffled_data = _BufferedShuffleIterator(data, 100)
    checkpointed = list(islice(shuffled_data, 10000-10))

    checkpoint = shuffled_data.__getstate__()
    data = NativeCheckpointableIterator(iter(range(data_size)))
    shuffled_data = _BufferedShuffleIterator(data, 100, 42)
    shuffled_data.__setstate__(checkpoint)
    checkpointed += list(shuffled_data)

    assert checkpointed == not_checkpointed
    print("passed")

    # TODO: make this a real test
    random = Random()
    for i in range(20):
        # random sequence lengths to for testing different configurations
        test_source_length        = random.randrange(5,25)
        test_first_output_length  = random.randrange(5,25)
        test_second_output_length = random.randrange(5,25)
        # source
        test_source = range(test_source_length)
        reader = _InfinitePermutationIterator(test_source, seed=i)
        # fetch a first sequence
        items0 = list(islice(reader, test_first_output_length))
        print('items0', items0)
        # fetch a second sequence
        checkpoint = reader.__getstate__()
        items1a = list(islice(reader, test_second_output_length))
        print('items1a', items1a)
        # fetch that second sequence again via checkpointing
        reader.__setstate__(checkpoint)
        items1b = list(islice(reader, test_second_output_length))
        print('items1b', items1b)
        # must be the same
        assert items1a == items1b
