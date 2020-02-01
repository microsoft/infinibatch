import gzip
import itertools
from itertools import islice
import os
from random import Random
from typing import Union, Iterable, Iterator, List, Any, Callable, Optional, Generator
import copy


# simple mechanism to create a class and class instance on the fly
# e.g. x = Record(a=13, b=42) then use x.a and x.b
# This mimics C#'s named tuple types (and is very different from Python's namedtuple which creates a type not class instances).
class Record:
    def __init__(self, **args_dict):
        self.__dict__.update(args_dict)
    def copy(self):
        return Record(**dict(self.__dict__.items()))
    # @TODO: make this immutable, rename back to Record, and get rid of copy(). Mutability currently only used in infinite_permutation_iterator()


class _ICheckpointIterator:  # @TODO: Can rename away the I- once done. This makes it easier during development
    def __next__(self):
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __getstate__(self):
        raise NotImplementedError()

    def __setstate__(self, checkpoint):
        raise NotImplementedError()


class _InfinitePermutationIterator(_ICheckpointIterator):  # ...how to say in Python: "implements interfaces Iterator[Any], Checkpointing"
    """
    Infinitely generates permutations of the items in the given iterable.

    Unlike most classes here, this one loads all items into RAM. For example, this is used
    for randomizing the pathnames of data blocks read by _IterableChunkedData.

    Arguments:
    iterator -- input iterator
    seed -- random seed used for shuffling (or None)
    """
    # constructor arguments
    _original_items: List[Any]  # note: in this case, the source iterator is not checkpointable, hence we must instead keep a full copy
    _seed: Optional[int]

    # output iterator that generates our output sequence
    _generator: Iterator[Any]

    # iteration state. This is returned when requesting the checkpoint, and restored when resetting to checkpoint.
    _random_state: Any
    _item_count: int

    def __init__(self, items: Iterator[Any], seed: Optional[int]):
        # keep arguments for iter_from_checkpoint
        self._original_items = list(items)  # keep a local copy, since items is an iterator
        self._seed = seed
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
    def __getstate__(self):
        return Record(random_state = self._random_state,  # state of random generator before generating the current shuffling of the sequence
                      item_count   = self._item_count,    # how many items have already been served from the current shuffling
                      nested_state = None)                # state of underlying iterator (not present in this example)

    def __setstate__(self, from_checkpoint):
        # set iteration state. Do this outside the generator below in case __getstate__() is called before ever iterating
        self._random_state = from_checkpoint.random_state if from_checkpoint else None
        self._item_count   = from_checkpoint.item_count   if from_checkpoint else 0
        # We define the iteration itself as a generator for ease of implementation.
        # We could as well just have used an explicit state machine represented by class members.
        def _create_generator():
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
                random.shuffle(shuffled_items)
                shuffled_iterator = iter(shuffled_items)
                # skip initial items when restarting from checkpoint
                if skip_to_checkpoint:
                    for i in range(skip_to_checkpoint):
                        next(shuffled_iterator)
                    self._item_count += skip_to_checkpoint
                    skip_to_checkpoint = 0  # done skipping
                # main inner loop over items
                for item in shuffled_iterator:
                    self._item_count += 1  # record how many items we have served from this pass over the items
                    yield item
        self._generator = _create_generator()


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


class NativeIterator(_ICheckpointIterator):
    def __init__(self, iterator: Iterator[Any]):
        self._iterator = iterator
        self._consumed_items = 0

    def __next__(self):
        item = next(self._iterator)
        self._consumed_items += 1
        return item

    def __getstate__(self) -> List:
        return [self._consumed_items]

    def __setstate__(self, checkpoint):
        self._consumed_items = checkpoint[-1]
        for _ in range(self._consumed_items):
            next(self._iterator)


class BufferedShuffleIterator(_ICheckpointIterator):
    def __init__(self, input_iterator: _ICheckpointIterator, buffer_size: int, seed: int = 0):
        self._input_iterator = input_iterator
        self._buffer = [None for _ in range(buffer_size)]  # maybe do this lazily?   --Yes, since user may set state immediately, then this is not needed here
        self._random = Random(seed)
        self._generator = self._create_generator()  # @TODO: call __setstate__ instead

    def __next__(self):
        return next(self._generator)

    def _create_generator(self):
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
            yield self._buffer.pop()

    def __getstate__(self):
        #previous_checkpoint = self._input_iterator.__getstate__()  # @TODO: let's not assume that the underlying iterator's checkpoint is a list
        #local_checkpoint = [copy.deepcopy(self._buffer), self._random.getstate()]
        #previous_checkpoint.append(local_checkpoint)
        return (self._input_iterator.__getstate__(),
                (copy.deepcopy(self._buffer),
                 self._random.getstate()))

    def __setstate__(self, checkpoint):
        self._input_iterator.__setstate__(checkpoint[0])
        local_checkpoint = checkpoint[1]
        self._buffer = local_checkpoint[0]
        self._random.setstate(local_checkpoint[1])  # @TODO: better recreate the generator
        # @BUGBUG?: Does this handle the flush part?


class _IterableBufferedShuffler:  # @TODO: we should next replace this by BufferedShuffleIterator above
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
            chunks = _InfinitePermutationIterator(self._chunk_file_paths, self._seed)
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



if __name__ == '__main__':
    data_size = 10**5

    data = NativeIterator(iter(range(data_size)))
    shuffled_data = BufferedShuffleIterator(data, 100)
    not_checkpointed = list(shuffled_data)

    data = NativeIterator(iter(range(data_size)))
    shuffled_data = BufferedShuffleIterator(data, 100)
    checkpointed = list(islice(shuffled_data, 10000-10))

    checkpoint = shuffled_data.__getstate__()
    data = NativeIterator(iter(range(data_size)))
    shuffled_data = BufferedShuffleIterator(data, 100, 42)
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
