from abc import abstractmethod
import collections
import copy
import gzip
from itertools import cycle, islice
import os
from queue import Full, Queue
from random import Random
from threading import Thread
from typing import Any, Callable, Dict, Generator, Iterable, Iterator, List, Optional, Tuple, Union


from .closablequeue import ClosableQueue, ClosedException


"""
infinibatch -- A library of checkpointable iterators for randomized data loading of massive data sets
in deep-neural-network training.

Features:

  * support for corpora much larger than fit into RAM
  * hierarchical block+sentence-level randomization over the whole corpus, different randomization in each epoch
  * only load the data that is needed
  * very fast start-up time (does not need to read full corpus)
  * only requires the most basic of data preparation (e.g. no indexing)
  * for multi-GPU, only load what the respective GPU needs
  * 100% accurate check-pointing, restore from checkpoint should not read all data up to the checkpoint
  * support automatic bucketed batching with dynamic batch sizes
  * pre-fetching thread
  * composable, as to support for complex batching, e.g. negative samples from multiple documents
"""


# TODO for next release:
#  - implement new version of BufferedShuffleIterator that has smaller checkpoints
#  - modify ChunkedReadlinesIterator to also work on uncompressed data, or even more general data formats
#  - add type checks to guarantee that input iterators are checkpointable
#  - change all convenience functions back to true classes, using a wrapper class

# TODO later:
# - make iterator pipeline work for streaming data


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

    def __getstate__(self) -> Dict:  # implementation of pickle Protocol
        return self.getstate()

    def __setstate__(self, checkpoint: Optional[Dict]):
        self.setstate(checkpoint)

    @abstractmethod
    def getstate(self) -> Dict:
        pass

    @abstractmethod
    def setstate(self, checkpoint: Optional[Dict]):
        pass

    @abstractmethod
    def __next__(self):
        pass


class NativeCheckpointableIterator(CheckpointableIterator):
    """
    Simple checkpointable wrapper around native Python iterable.
    This version just replays the iterator all the way to the checkpoint, which will
    make it inefficient for some important use cases.

    Warning: This class cannot be used with Iterators (as opposed to Iterables), which have an __iter__ function that simply returns self, but does not reset.
    """
    def __init__(self, iterable: Iterable):
        # check whether iterable is iterable or iterator:
        # if the variable iterable contains an iterator, the function __iter__ returns self
        # if the variable iterable is an actual iterator, it should not return self
        if iter(iterable) is iterable:  
            raise ValueError('It looks like you are passing an iterator instead of an iterable. This is not supported and can cause undefined behavior when used with checkpointing.')
        self._input_iterable = iterable
        self.setstate(None)

    def getstate(self) -> Dict:
        return {'consumed_items': self._consumed_items}

    def setstate(self, checkpoint: Optional[Dict]):
        self._iterator = iter(self._input_iterable)
        self._consumed_items = _advance_iterator(self._iterator, checkpoint['consumed_items']) if checkpoint is not None else 0

    def __next__(self):
        item = next(self._iterator)  # call this before increasing _consumed_items to correctly handle the case when a StopIteration exception is thrown
        self._consumed_items += 1
        return item


class InfinitePermutationIterator(CheckpointableIterator):
    """
    Infinitely generates permutations of the items in the given iterable.

    Unlike most classes here, this one loads all items into RAM. For example, this is used
    for randomizing the pathnames of data blocks read by ChunkedReadlinesIterator.
    """
    def __init__(self, source_iterator: Iterator, seed: Optional[int]=None, shuffle: bool=True, num_instances: int=1, instance_rank: int=0):
        """
        Args:
            source_iterator: input iterator
            seed: random seed used for shuffling (or None)
            shuffle: set False to bypass the shuffling. Then this is just a checkpointed version of itertools.cycle(). (Default: True)
            num_instances: number of instances of this dataset. Meant for use with multi-process data loading, e.g., in distributed training.
            instance_rank: rank of this instance of the dataset. Meant for use with multi-process data loading, e.g., in distributed training.
        """
        self._source_items = list(source_iterator)  # keep a local copy
        self._shuffle = shuffle
        self._seed = seed
        self._num_instances = num_instances
        self._instance_rank = instance_rank
        self.setstate(None)

    def getstate(self) -> Dict:
        return {'random_state': self._random_state,  # state of random generator before generating the current shuffling of the sequence
                'item_count':   self._item_count}    # how many items have already been iterated over in the current shuffling

    def setstate(self, checkpoint: Optional[Dict]):
        # set iteration state. Do this outside the generator below in case getstate() is called before ever iterating
        self._random_state = checkpoint['random_state'] if checkpoint else None
        self._item_count   = checkpoint['item_count']   if checkpoint else 0
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
                shuffled_items = self._source_items[:]  # note: if underlying iterator is checkpointable, use setstate(checkpoint['nested_state']) on it
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
    """
    Projects each element of a source sequence to a sequence and flattens the resulting sequences into one sequence.
    """
    def __init__(self, source_iterator: CheckpointableIterator, collection_selector: Callable[[Any], Iterable]):
        """
        Args:
            collection_selector: user callback that maps an item into an Iterable, whose items will be yielded.
                                 The returned Iterable is used only once. Hence, it is also allowed to
                                 return self-iterables, such as iterators and generator expressions.
            source_iterator: iterable of paths to chunk files
        """
        self._source_iterator: CheckpointableIterator = source_iterator
        self._collection_selector: Callable[[Any], Iterable] = collection_selector
        self.setstate(None)

    def getstate(self) -> Dict:
        return {'nested_state': self._input_state,
                'item_index': self._flattened_item_index}

    def setstate(self, checkpoint: Optional[Dict]):
        self._input_state           = checkpoint['nested_state'] if checkpoint else None
        self._flattened_item_index  = checkpoint['item_index']   if checkpoint else 0
        self._source_iterator.setstate(self._input_state)
        def _generate():
            skip_to_checkpoint = self._flattened_item_index
            # main loop over source source_items
            for source_item in self._source_iterator:
                data = iter(self._collection_selector(source_item))
                self._flattened_item_index = 0
                if skip_to_checkpoint:
                    #print("Skipping to index", skip_to_checkpoint, file=sys.stderr)
                    self._flattened_item_index += _advance_iterator(data, skip_to_checkpoint)
                    skip_to_checkpoint = 0
                # main loop over lines
                for item in data:
                    self._flattened_item_index += 1
                    yield item
                self._input_state = self._source_iterator.getstate()
        self._iterator = _generate()

    def __next__(self):
        return next(self._iterator)


# @TODO: Can we seamlessly support UCS-2 files as well? C# can auto-detect. Does Python have such a facility?
def ChunkedReadlinesIterator(chunk_file_paths: CheckpointableIterator):
    """
    Reads text lines from zipped chunk files whose names are provided by an iterator.

    Args:
        chunk_file_paths: CheckpointableIterator of paths to chunk files
    """
    def readlines_from_zipped(textfile_path: str) -> Iterable[str]:
        #print("Reading chunk file", textfile_path, file=sys.stderr)
        with gzip.open(textfile_path, 'rt', encoding='utf-8') as f:
            return iter(f.read().splitlines())
    return SelectManyIterator(source_iterator=chunk_file_paths, collection_selector=readlines_from_zipped)


class BufferedShuffleIterator(CheckpointableIterator):
    """
    Shuffles given iterable using a limited buffer.
    """
    def __init__(self, source_iterator: CheckpointableIterator, buffer_size: int, seed: int = 0):
        """
        Args:
            source_iterator: checkpointable iterator or restartable iterable over input items to shuffle
            buffer_size: size of the buffer in number of items used for shuffling
            seed: random seed used for shuffling (or None)
        """
        self._source_iterator = source_iterator
        self._buffer = [None for _ in range(buffer_size)]  # maybe do this lazily?   --Yes, since user may set state immediately, then this is not needed here
        self._random = Random(seed)
        self.setstate(None)

    def getstate(self) -> Dict:
        return {'nested_checkpoint': self._source_iterator.getstate(),
                'buffer':            copy.deepcopy(self._buffer),
                'random_state':      self._random.getstate()}

    def setstate(self, checkpoint: Optional[Dict]):
        if checkpoint:
            self._source_iterator.setstate(checkpoint['nested_checkpoint'])
            self._buffer = checkpoint['buffer']
            self._random.setstate(checkpoint['random_state'])
            # @TODO: Can we add a comment how the flush part is handled?
        else:
            self._source_iterator.setstate(None)
        self._generator = self._generate()

    def _generate(self) -> Iterator:
        # shuffle data with a buffer:
        # this is similar to what the Fisher-Yates shuffle does,
        # but modified to run with a constant-size buffer
        # see https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
        # this was inspired by an algorithm implemented in Kaldi
        # see https://kaldi-asr.org/doc/nnet-shuffle-egs_8cc.html
        for item in self._source_iterator:
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


class MapIterator(CheckpointableIterator):
    """
    Applies given tranform to each data item
    """
    def __init__(self, source_iterator: CheckpointableIterator, transform: Callable[[str],Any]=None):
        """
        Args:
            source_iterator: checkpointable iterator
            transform: function to be applied to each data item
        """
        self._source_iterator = source_iterator
        self._transform = transform

    def getstate(self) -> Dict:
        return self._source_iterator.getstate()

    def setstate(self, checkpoint: Optional[Dict]):
        self._source_iterator.setstate(checkpoint)

    def __next__(self):
        return self._transform(next(self._source_iterator))


class ZipIterator(CheckpointableIterator):
    """
    Zips items from all given iterators, like the Python standard function zip().

    Like Python's build-in zip(), the iteration stops when the shortest input iterable is exhausted.
    """
    def __init__(self, *source_iterators: CheckpointableIterator):
        """
        Args:
            source_iterators: list of iterators to zip, item by item
        """
        self._source_iterators: List[CheckpointableIterator] = source_iterators

    def getstate(self) -> Dict:
        return {'input_states': tuple(iterator.getstate() for iterator in self._source_iterators)}

    def setstate(self, checkpoint: Optional[Dict]):
        if checkpoint is None:
            for iterator in self._source_iterators:
                iterator.setstate(None)
        else:
            for iterator, state in zip(self._source_iterators, checkpoint['input_states']):
                iterator.setstate(state)

    def __next__(self):
        res = []  # (note: can't use a generator expression, as it gets confused when a next() call raises StopIteration)
        for iterator in self._source_iterators:
            res.append(next(iterator))
        return tuple(res)


# @TODO: The yield makes a (shallow) copy of the window, which has complexity O(width * length). In some cases,
#        we don't actually need to consume all items in the window. Hence, to make this faster, we should use
#        double-buffering and return a slice view (which we'd have to write).
class WindowedIterator(CheckpointableIterator):
    """
    Yields 'width' consecutive items in a sliding window.

    E.g. [1, 2, 3 4, 5, 6] with width = 3 will yield
    [(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6)]
    """
    def __init__(self, source_iterator: CheckpointableIterator, width: int):
        """
        Args:
            source_iterator: checkpointable input iterators
        """
        self._source_iterator: CheckpointableIterator = source_iterator
        self._width: int = width
        self.setstate(None)

    def getstate(self) -> Dict:
        return {'input_state': self._input_state,  # state for first item in FIFO
                'item_index':  self._item_index}   # index of next item to serve

    def setstate(self, checkpoint: Optional[Dict]):
        self._input_state = checkpoint['input_state'] if checkpoint else None
        self._item_index  = checkpoint['item_index']  if checkpoint else 0
        self._source_iterator.setstate(self._input_state)
        self._generator = self._generate()

    def _fifo_slice(self, i):  # returns a window into the FIFO beginning at i
        # @TODO: for efficiency, make this a slice view
        return tuple(self._fifo[i:i + self._width])

    def _generate(self) -> Iterator:
        self._input_state = self._source_iterator.getstate()
        self._fifo = list(islice(self._source_iterator, self._width))
        # we do this in overlapping blocks of length 2*width, for easier checkpointing and potential efficiency
        while len(self._fifo) == self._width:
            # we got 'width' items; append another 'width' (or less if at end)
            next_input_state = self._source_iterator.getstate()
            self._fifo.extend(islice(self._source_iterator, self._width))
            # now serve all positions in first half (last = width - 1). If at end, then limit accordingly.
            last = min(self._width - 1, len(self._fifo) - self._width)
            while self._item_index <= last:
                window = self._fifo_slice(self._item_index)
                self._item_index += 1
                yield window
            # drop all we just served; if < width left, we have hit the end
            self._fifo = self._fifo[last + 1:]    # Note: This must be a new list, since the old might still be in a slice view.
            self._input_state = next_input_state  # this reflects now the first element in the FIFO 
            self._item_index = 0

    def __next__(self):
        return next(self._generator)


class RandomIterator(CheckpointableIterator):
    """
    Iterator to generate uniformly distributed random numbers in the interval [0,1).
    Very similar to Random.random(), except that random numbers are
    obtained via next().
    """
    def __init__(self, seed: Optional[int]=None):
        """
        Args:
            seed: Random seed.
        """
        self._random: Random = Random()
        if seed is not None:
            self._random.seed(seed)

    def getstate(self) -> Dict:
        return {'random_state': self._random.getstate()}

    def setstate(self, checkpoint: Optional[Dict]):
        self._random.setstate(checkpoint['random_state'] if checkpoint else None)

    def __next__(self):
        return self._random.random()


# It is not clear whether there is much value in this one. Let's leave it commented-out
# for a while, then decide whether to delete or uncomment it?
#def SamplingMapIterator(input_iterator: CheckpointableIterator, sampling_transform: Callable[[float,Any],Any], seed: Optional[int]=None):
#    """
#    Iterates over a checkpointable iterator and invokes a user-supplied transform function
#    as sampling_transform(rand_val, item), where rand_val is a random number in [0,1).
#
#    Args:
#        sampling_transform: a callable with signature (rand_val, item)
#        seed: Random seed.
#    """
#    r = RandomIterator(seed)
#    i = ZipIterator(r, input_iterator)  # generates tuples (random number, input item)
#    def _wrapped_transform(arg: Tuple[float, Any]) -> Any:  # invokes user's transform function with the tuple members as the arguments
#        randval, item = arg
#        return sampling_transform(randval, item)
#    return MapIterator(i, _wrapped_transform)


class RecurrentIterator(CheckpointableIterator):
    """
    Iterates statefully over a step function. The step function accepts a state and a new item,
    and returns a new state and an output item, which is yielded.
    """
    def __init__(self, source_iterator: CheckpointableIterator, step_function: Callable[[Any,Any], Tuple[Any,Any]], initial_state: Any = None):
        """
        Args:
            source_iterator: checkpointable iterator to recur over
            step_function: user-supplied function with signature step_function(state, item) -> (new_state, output)
            initial_state: initial state to be passed to the step_function upon first invocation
        """
        self._source_iterator: CheckpointableIterator = source_iterator
        self._step_function: Callable[[Any,Any], Tuple[Any,Any]] = step_function
        self._initial_state: Any = initial_state
        self.setstate(None)
    
    def getstate(self):
        return {'recurrent_state': self._recurrent_state,
                'source_state':    self._source_iterator.getstate()}
    
    def setstate(self, checkpoint):
        self._recurrent_state = checkpoint['recurrent_state'] if checkpoint else self._initial_state
        self._source_iterator.setstate(checkpoint['source_state'] if checkpoint else None)
        def _generate():
            for item in self._source_iterator:
                self._recurrent_state, output = self._step_function(self._recurrent_state, item)
                yield output
        self._iterator = _generate()

    def __next__(self):
        return next(self._iterator)


def SamplingRandomMapIterator(source_iterator: CheckpointableIterator, transform: Callable[[Random,Any],Any], seed: Optional[int]=None):
    """
    An iterator that calls a transform function on each item, while also passing a checkpointed
    random generator.

    Args:
        source_iterator: checkpointable iterator to recur over
        step_function: user-supplied function with signature step_function(random, item) -> result_item
        seed: random seed
    """
    _random = Random()
    if seed:
        _random.seed(seed)
    def _step_function(state, item):
        _random.setstate(state)
        output = transform(_random, item)
        return _random.getstate(), output
    return RecurrentIterator(source_iterator, _step_function, initial_state=_random.getstate())


class PrefetchIterator(CheckpointableIterator):
    """
    An iterator prefetching data into a buffer on a seperate thread to smooth out IO latency.

    Args:
        source_iterator: checkpointable iterator to recur over
        buffer_size: size of the queue between the threads
    """
    def __init__(self, source_iterator: CheckpointableIterator, buffer_size: int=1000):
        self._source_iterator: CheckpointableIterator = source_iterator
        self._buffer_size: int = buffer_size
        self._queue: Optional[ClosableQueue] = None
        self._thread: Optional[Thread] = None
        self.setstate(None)
        
    def getstate(self) -> Dict:
        return {'source_state': self._source_state,
                'item_offset' : self._item_offset  }

    def setstate(self, checkpoint: Optional[Dict]):
        if self._thread is not None:  # if there is a prefetching thread running, close the queue and wait for the thread to terminate
            assert self._queue is not None
            self._queue.close()
            self._thread.join()
        
        self._source_state = checkpoint['source_state'] if checkpoint is not None else None
        self._item_offset = checkpoint['item_offset'] if checkpoint is not None else 0

        self._source_iterator.setstate(self._source_state)

        self._queue = ClosableQueue(maxsize=self._buffer_size)  # clear queue
        # make thread daemonic so it is killed when the main program terminates
        self._thread = Thread(target=self._prefetch_thread_fn, args=(self._source_iterator, self._item_offset, self._buffer_size, self._queue), daemon=True)
        self._thread.start()

    @staticmethod
    def _prefetch_thread_fn(source, item_offset, buffer_size, queue):  # behavior of the prefetching thread, only call from that thread!
        _advance_iterator(source, item_offset)  # skip to checkpoint

        while True:
            try:
                item = next(source)
            except StopIteration:
                queue.close()
                return
            
            if item_offset == buffer_size - 1:  # send a new source state a the END of each window of length _buffer_size
                source_state = source.getstate()  # this is the state for retrieving the NEXT element, i.e. the first element of the next buffer
                item_offset = 0
            else:
                source_state = None
                item_offset += 1
            msg = (item, source_state)

            try:
                queue.put(msg)
            except ClosedException:
                return

    def __next__(self):
        try:
            msg = self._queue.get()
        except ClosedException:
            raise StopIteration

        item, prefetch_source_state = msg
        if prefetch_source_state is not None:
            assert self._item_offset == self._buffer_size - 1  # we expect a new source state at then END of each window of length _buffer_size
            self._source_state = prefetch_source_state
            self._item_offset = 0
        else:
            self._item_offset = self._item_offset + 1
            assert self._item_offset < self._buffer_size
        return item  # for debugging, its useful to return msg instead of item


class BucketedReadaheadBatchIterator(CheckpointableIterator):
    """
    Iterates over items from a checkpointable iterator and groups items of similar length into batches.

    The algorithm reads a head a certain number of lines (e.g. 10 million), sorts them by
    length, and them groups them into batches from start to end. The sort is stable, such
    that prior randomization is not undone (except for the length grouping). The batch size
    is dynamic, and determined by a user-provided callback.

    This is based on Marian NMT's BatchGenerator.
    """
    # @TODO: We had agreed to remove the explicit member declarations, and instead implicitly declare them in __init__ upon assignment.
    # parameters
    _key: Callable[[Any], Any]
    _batch_size: Union[int,Callable[[Any], int]]
    _read_ahead: int

    # state
    _source_iterator: Iterator[Any]   # iterator into _source
    _random: Random             # random generator
    _source_exhausted: bool    # set to True once we hit StopIteration on source
    _batch_iter: Iterator[Any]  # iterator into current set of batches
    _input_state: Dict    # state of input before reading the current set of batches
    _num_served: int            # number of batches served from the current set of batches

    def __init__(self, source_iterator: CheckpointableIterator, read_ahead: int, key: Callable[[Any], Any], batch_size: Union[int,Callable[[Any], int]], shuffle: bool=True, seed: Optional[int]=None):
        """
        Args:
            source_iterator: The data set that is read from. Typically this is an infinite source.
            read_ahead: Number of items to fetch ahead for grouping purposes.
            key: User-provided callback to define how data is sorted for purpose of batching.
            batch_size: Batch size in number of items. Either an integer or a callback to determine batch size for a given first batch item.
            shuffle: Pass False to not randomize the batches. (default: True)
            seed: Random seed for batch shuffling.
        """
        # keep arguments
        self._key = key
        self._batch_size = batch_size
        self._read_ahead = read_ahead
        # initialize state
        self._random = None
        if shuffle:
            self._random = Random()
            if seed is not None:
                self._random.seed(seed)
        self._source_iterator = iter(source_iterator)
        self.setstate(None)

    def getstate(self):
        return {'input_state':  self._input_state,
                'random_state': self._random_state,
                'num_served':   self._num_served}

    def setstate(self, checkpoint: Optional[Dict]):
        self._input_state  = checkpoint['input_state']  if checkpoint else None
        self._random_state = checkpoint['random_state'] if checkpoint else None
        self._num_served   = checkpoint['num_served']   if checkpoint else 0
        # checkpointing: restore to start of current set of batches
        self._source_iterator.setstate(self._input_state)
        if self._random_state:
            self._random.setstate(self._random_state)
        self._source_exhausted = False
        def _generate():
            skip_to_checkpoint = self._num_served
            source_exhausted = False
            while not source_exhausted:
                # prefetch the readahead buffer
                self._input_state = self._source_iterator.getstate()
                self._random_state = self._random.getstate() if self._random else None
                items = list(islice(self._source_iterator, self._read_ahead))
                source_exhausted = (len(items) < self._read_ahead)
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