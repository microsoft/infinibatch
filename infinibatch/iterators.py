"""
## Overview

This part of the documentation covers the __advanced usage__ of Infinibatch by assembling __custom data loading pipelines__.
Before you continue, please go through the tutorial on the top-level of the documentation of the `infinibatch` module.

Two of the main features of Infinibatch are __lazy evaluation__ through the use of __iterators__
and built-in support for __checkpointing__.
In this section, we give an introduction to these features and the basic usage of the Infinibatch iterator library.


### Iterators

As a Python programmer, you are probably familiar with the concept of iterators.
According to the [Python documentation](https://docs.python.org/3.5/glossary.html#term-iterator),
an iterator is an object representing a stream of data,
and repeated calls to the iterator's `__next__()` method (or passing it to the built-in function `next()`)
return successive items in the stream.
It is important not to confuse an [iterator](https://docs.python.org/3.5/glossary.html#term-iterator)
with an [iterable](https://docs.python.org/3.5/glossary.html#term-iterable).
For more information on this subject, please follow the links above.

The Python standard library contains a module of iterators called `itertools`
that bears some resembles to Infinibatch.
Infinibatch differs from `itertools` in two ways:

1. Infinibatch provides iterators specifically for the purpose of creating __randomized batches of data for machine learning__.
2. All iterators in Infinibatch support __checkpointing__ (see the following section).

Infinibatch iterators are not directly compatible with itertools due to the checkpointing requirement.

Infinibatch enables you to build complex data loaders by combining iterators from this module into a pipeline.
To give you a high-level idea of how this is works, we provide a very simple example.
Note that this example is completely artificial and does not solve any useful task.
Its only purpose is to demonstrate the behavior of a pipeline of iterators.
We provide a more realistic example in a later section.

First, we create a small test data set.
>>> dataset = list(range(6))  # 0, 1, 2, 3, 4, 5

We can turn this data set into an Infinibatch iterator by wrapping it in a `NativeCheckpointableIterator`.
>>> it = NativeCheckpointableIterator(dataset)  # 0, 1, 2, 3, 4, 5

We can then transform the data items using a `MapIterator`,
which applies a given function to each individual data item.
For example, we can multiply each data item by 2.
>>> it = MapIterator(it, lambda n: 2 * n)  # 0, 2, 4, 6, 8, 10

We can restructure the data set by batching together pairs of data items into lists using a `FixedBatchIterator`.
>>> it = FixedBatchIterator(it, batch_size=2)  # [0, 2], [4, 6], [8, 10]

Using another `MapIterator`, we can reduce each of these lists to its second element.
>>> it = MapIterator(it, lambda l: l[1])  # 2, 6, 10

Finally, we can use the resulting iterator `it` just like any standard Python iterator.
```py
>>> for item in it:
...     print(item)
2
6
10

```

By using iterators, Infinibatch operates in a __lazy__ fashion:
It generally doesn't apply operations to an entire data set at once,
but rather operates on individual data items on-the-fly as they are consumed.
When used correctly, this allows Infinibatch to have a low start-up time and low memory overhead.
For more detail on this, please consult the section on performance considerations below.


### Checkpointing

The main features that sets Infinibatch iterators apart from standard Python iterators is that they support __checkpointing__.
A checkpoint encapsulates the internal state of an entire pipeline of iterators at a specific point while iterating through a data set.
Once you retrieve a checkpoint, you can later use it to reset the pipeline of iterators to the exact state it was in
when the checkpoint was created.
Checkpoints can easily be serialized and stored to disk using [Pythons `pickle` module](https://docs.python.org/3.5/library/pickle.html).
Infinibatch's checkpointing feature is particularly useful when you're training large deep neural network models over days or weeks,
and you want to make sure that, in case your training is interrupted for any reason, __you can pick up your training exactly where you left off__.

The checkpointing interface consists of two functions `getstate` and `setstate` that are defined in `CheckpointableIterator`,
the common base class of all iterators in this module.
As the names suggest `getstate` returns a checkpoint object that represents the state of a pipeline at the time the function is called,
and 'setstate' receives a checkpoint object to reset the state of a pipeline.
`setstate` also accepts `None`, which resets a pipeline to the __beginning__ of the iteration,
i.e. the state of the pipeline immediately after its construction.

It is important to realize that __a checkpoint represents the state of a complete pipeline of iterators__.
If you have a pipeline consisting of a sequence of iterators, you only have to call `getstate` on the __last__ iterator in the sequence
to capture the state of the entire pipeline.
Internally, this is achieved by recursive calls that traverse the entire data loading pipeline to collect the state of every iterator in it.
Similarly, when you want to reset a pipeline to a previous state, you only have to call `setstate` on the __last__ iterator in the pipeline.


To demonstrate this, we recreate the pipeline from the previous section.
>>> dataset = list(range(6))  # 0, 1, 2, 3, 4, 5
>>> it = NativeCheckpointableIterator(dataset)  # 0, 1, 2, 3, 4, 5
>>> it = MapIterator(it, lambda n: 2 * n)  # 0, 2, 4, 6, 8, 10
>>> it = FixedBatchIterator(it, batch_size=2)  # [0, 2], [4, 6], [8, 10]
>>> it = MapIterator(it, lambda l: l[1])  # 2, 6, 10

Since `it` behaves just like a standard Python iterator, we can call `next` to retrieve its first element.
>>> next(it)
2

We can now call `getstate` on `it` (which is the last `MapIterator` in the pipeline)
to get a checkpoint of the internal state of the entire data loading pipeline.
>>> checkpoint = it.getstate()

Note that the checkpoint represents the internal state of the pipeline after the data item `2` has been retrieved.
Using the checkpoint, we can always return to this __exact__ point in the data set.
To show this, let's exhaust the iterator by casting it to a list.
>>> list(it)
[6, 10]

Since the iterator is now exhausted, calling `next` raises a `StopIteration` exception.
```
>>> next(it)
Traceback (most recent call last):
    ...
StopIteration

```

We can now reset the pipeline to the checkpoint using `setstate`.
>>> it.setstate(checkpoint)

This recovers the state of the pipeline after the data item `2` has been retrieved.
Thereby, we expect the next element to be `6`.
>>> next(it)
6


## Types of Iterators

This section provides a brief overview of the different types of iterators in Infinibatch.


### Classes and Factory Functions

Most iterators in this module are implemented as classes that inherit from the abstract base class `CheckpointableIterator`.
However, some iterators (such as the `BlockwiseShuffleIterator`) are simple combinations of other iterators.
These iterators are implemented as __factory functions__ that construct a pipeline of iterators
and return the last iterator in the pipeline.
For consistency with class-based iterators,
we name these factory function using CamelCase instead of the more pythonic use_of_underscores.

.. todo::
    We currently also have one factory function that actually looks like one: `create_source_iterator`.
    Provide a comment on this describing why that is.


### Source Iterators

There are three iterators that are intended to go at the __beginning__ of a data loading pipeline:

- `InfinitePermutationSourceIterator`:
This iterator accepts a list, shuffles it, and yields its elements.
It repeats this infinitely, shuffling the list after each pass.
Thereby, __this iterator is infinte and cannot be exhausted__.
This iterator is meant to be used as the first iterator in a training scenario
and supports splitting the data for multi-GPU training.
- `ChunkedSourceIterator`:
This iterator accepts a list and yields its elements.
It is meant to be used as the first iterator in an inference or validation scenario
and supports splitting the data for mult-GPU inference.
- `NativeCheckpointableIterator`:
This iterator wraps a Python iterable and makes it checkpointable.
It is mainly intended for demonstration and debugging purposes.


### Shuffling

.. todo:: Describe `BufferedShuffleIterator` and `BlockwiseShuffleIterator`.


### Batching, SelectMany, and Windowing

.. todo:: Describe `FixedBatchIterator`, `SelectManyIterator`, and `WindowedIterator`.


### Mapping

.. todo:: Describe `MapIterator`, `ParallelMapIterator`, `RecurrentIterator`, and `SamplingRandomMapIterator`.


### Other Iterators

.. todo:: Describe `ZipIterator`, `PrefetchIterator`, and `BucketedReadaheadBatchIterator`.


## Complete Example

.. todo::
    Give a more realistic example following, in broad strokes, the ChunkedDataset including:

    - use gzip chunks
    - training pipeline example
    - inference pipeline example
    - pipeline that can do both
    - etc.

## Performance Considerations

.. todo::
    Describe what parameters influence performance measures such as memory usage and start-up time.
"""

from abc import abstractmethod
import collections
import copy
import gzip
from itertools import cycle, islice
import math
import multiprocessing as python_multiprocessing
import os
from random import Random
from typing import Any, Callable, Dict, Generator, Iterable, Iterator, List, Optional, Tuple, Union, cast


# TODO for next release:
#  - benchmark the accuracy when using BlockwiseShuffleIterator vs. the BufferedShuffleIterator
#  - change all convenience functions back to true classes, using a wrapper class

# TODO later:
# - make iterator pipeline work for streaming data

def _advance_iterator(iterator: Iterator, n: int):
    """ Little helper to advance an iterator by n items """
    for i in range(n):
        try:
            next(iterator)
        except StopIteration:
            raise RuntimeError('Trying to advance iterator by {} but iterator raised StopIteration exception on call to next with index {}.'.format(n, i))
    return n


class CheckpointableIterator(collections.abc.Iterator):
    """
    Abstract base class that defines the interface for checkpointing.

    The interface (getstate, setstate) is inspired by Python's random package.
    """
    def __iter__(self) -> 'CheckpointableIterator':
        """
        Returns an iterator over all iterable elements.

        Args:
            self: (todo): write your description
        """
        return self

    @abstractmethod
    def getstate(self) -> Dict:
        """
        Get checkpoint of current state of iterator

        In a pipeline of iterators, this function __recursively__ calls itself on the preceeding iterator
        and includes the gathered information in the returned checkpoint.
        Thereby, to obtain a checkpoint of the state of an entire pipeline of iterators
        you only have to call this function on the __last__ iterator in the pipeline.
        A checkpoint is represented as a `dict`,
        but the caller should treat a checkpoint as an opaque object
        and not make any assumptions about the existence or meaning of the `dict` entries.
        """
        pass

    @abstractmethod
    def setstate(self, checkpoint: Optional[Dict]):
        """
        Set state of iterator to given checkpoint

        In a pipeline of iterators, this function __recursively__ calls itself on the preceeding iterator.
        Thereby, to set the state of an entire pipeline of iterators to a given checkpoint
        you only have to call this function on the __last__ iterator in the pipeline.

        Args:
            checkpoint: Checkpoint that should be used to reset the state of the iterator (or pipeline).
                        If this is __None__, the state of the iterator (or pipeline) is reset to the initial
                        state immediately after construction.
        """
        pass

    def __getstate__(self) -> Dict:  # implementation of pickle Protocol
        """
        Returns the state of the current state.

        Args:
            self: (todo): write your description
        """
        return self.getstate()

    def __setstate__(self, checkpoint: Optional[Dict]):
        """
        Sets the checkpoint.

        Args:
            self: (todo): write your description
            checkpoint: (bool): write your description
        """
        self.setstate(checkpoint)

    @abstractmethod
    def __next__(self):
        """
        Returns the next callable.

        Args:
            self: (todo): write your description
        """
        pass


class NativeCheckpointableIterator(CheckpointableIterator):
    """
    Simple wrapper class that turns a Python Iterable into a CheckpointableIterator

    When calling setstate on this class, it simply replays the iterator all the way to the checkpoint one element at a time,
    which makes it generally inefficient.

    Warning: This class cannot be used with Iterators (as opposed to Iterables), which have an `__iter__` function that simply returns self, but does not reset.
    """
    def __init__(self, iterable: Iterable):
        """
        Initialize an iterable.

        Args:
            self: (todo): write your description
            iterable: (todo): write your description
        """
        # check whether iterable is iterable or iterator:
        # if the variable iterable contains an iterator, the function __iter__ returns self
        # if the variable iterable is an actual iterator, it should not return self
        if iter(iterable) is iterable:
            raise ValueError('It looks like you are passing an iterator instead of an iterable. This is not supported and can cause undefined behavior when used with checkpointing.')
        self._input_iterable = iterable
        self.setstate(None)

    def getstate(self) -> Dict:
        """
        Get the state of the state.

        Args:
            self: (todo): write your description
        """
        return {'num_items_yielded': self._num_items_yielded}

    def setstate(self, checkpoint: Optional[Dict]):
        """
        Sets the checkpoint.

        Args:
            self: (todo): write your description
            checkpoint: (bool): write your description
        """
        self._iterator = iter(self._input_iterable)
        self._num_items_yielded = _advance_iterator(self._iterator, checkpoint['num_items_yielded']) if checkpoint is not None else 0

    def __next__(self):
        """
        Return an iterator over the next item.

        Args:
            self: (todo): write your description
        """
        item = next(self._iterator)  # call this before increasing _num_items_yielded to correctly handle the case when a StopIteration exception is thrown
        self._num_items_yielded += 1
        return item


def create_source_iterator(source_items: List, train: bool=True, seed: Optional[int]=None, shuffle: bool=True, num_instances: int=1, instance_rank: int=0) -> CheckpointableIterator:
    """
    Creates an iterator for batches.

    Args:
        source_items: (str): write your description
        train: (bool): write your description
        seed: (int): write your description
        shuffle: (bool): write your description
        num_instances: (int): write your description
        instance_rank: (todo): write your description
    """
    if not train and shuffle:
        raise ValueError('shuffling is not supported when train=False')
    if train:
        return InfinitePermutationSourceIterator(source_items, seed=seed, shuffle=shuffle, num_instances=num_instances, instance_rank=instance_rank)
    else:
        return ChunkedSourceIterator(source_items, num_instances=num_instances, instance_rank=instance_rank)


def ChunkedSourceIterator(source_items: List, num_instances: int=1, instance_rank: int=0) -> CheckpointableIterator:
    """
    Cuts source list into chunks, one per instance, and serves out items in chunk corresponding to instance_rank

    This is a source iterator:
    It is meant to be used at the beginning of a data loading pipeline.
    As such, it takes a list as its source and not a CheckpointableIterator.

    Args:
        source_items: input list, must not be empty and must be small enough to fit into RAM entirely, ownership of the list and the data goes to the iterator, do not modify it!
        num_instances: number of instances of this iterator. Meant for use with multi-process data loading, e.g., in distributed training.
        instance_rank: rank of this instance of the iterator. Meant for use with multi-process data loading, e.g., in distributed training.
    """
    # heuristic: assuming blocks are all of the same size, math.ceil should give us the shortest makespan
    chunk_size = math.ceil(len(source_items) / num_instances)
    # this does not cause any out-of-bounds issues:
    # a slice with a start-index beyong the end of the list is empty,
    # and an end-index of a slice is capped at the end of the list
    chunk = source_items[instance_rank * chunk_size : (instance_rank + 1) * chunk_size]
    return NativeCheckpointableIterator(chunk)


class InfinitePermutationSourceIterator(CheckpointableIterator):
    """
    Infinitely generates permutations of the items in the given list.

    This is a source iterator:
    It is meant to be used at the beginning of a data loading pipeline.
    As such, it takes a list as its source and not a CheckpointableIterator.
    The given list is loaded completely into RAM.

    For example, this is used for randomizing the pathnames of data blocks read by ChunkedReadlinesIterator.
    """
    def __init__(self, source_items: List, seed: Optional[int]=0, shuffle: bool=True, num_instances: int=1, instance_rank: int=0):
        """
        Args:
            source_items: input list, must not be empty and must be small enough to fit into RAM entirely, ownership of the list and the data goes to the iterator, do not modify it!
            seed: random seed used for shuffling
            shuffle: set False to bypass the shuffling. Then this is just a checkpointed version of itertools.cycle(). (Default: True)
            num_instances: number of instances of this iterator. Meant for use with multi-process data loading, e.g., in distributed training.
            instance_rank: rank of this instance of the iterator. Meant for use with multi-process data loading, e.g., in distributed training.
        """
        self._source_items = source_items
        if not self._source_items:
            raise ValueError("InfinitePermutationIterator: source must not be empty")
        self._shuffle = shuffle
        self._seed = seed
        self._num_instances = num_instances
        self._instance_rank = instance_rank
        self.setstate(None)

    def getstate(self) -> Dict:
        """
        Return a random state of random values.

        Args:
            self: (todo): write your description
        """
        return {'random_state':      self._random_state,  # state of random generator before generating the current shuffling of the sequence
                'num_items_yielded': self._num_items_yielded}    # how many items have already been iterated over in the current shuffling

    def setstate(self, checkpoint: Optional[Dict]):
        """
        Yields the state.

        Args:
            self: (todo): write your description
            checkpoint: (bool): write your description
        """
        # set iteration state. Do this outside the generator below in case getstate() is called before ever iterating
        self._random_state      = checkpoint['random_state']      if checkpoint else None
        self._num_items_yielded = checkpoint['num_items_yielded'] if checkpoint else 0
        # We define the iteration itself as a generator for ease of implementation.
        # We could as well just have used an explicit state machine represented by class members.
        def _generate() -> Iterator:
            """
            Generate a sequence items.

            Args:
            """
            # create and reset random generator
            random = Random(self._seed)
            if self._random_state is not None:  # restore the random generator's state
                random.setstate(self._random_state)
            skip_to_checkpoint = self._num_items_yielded  # items to skip in order to advance to checkpoint
            # main outer loop for infinite passes over items (reshuffle before each pass)
            while True:
                # (re-)shuffle all items
                self._random_state = random.getstate()  # remember random state before shuffling
                self._num_items_yielded   = 0
                shuffled_items = self._source_items[:]  # note: if underlying iterator is checkpointable, use setstate(checkpoint['nested_state']) on it
                if self._shuffle:
                    random.shuffle(shuffled_items)
                shuffled_iterator = iter(shuffled_items)
                # skip initial items when restarting from checkpoint
                if skip_to_checkpoint:  # @TODO: find a way to abstract this more, so that we can plug it into the 'for' statement directly
                    self._num_items_yielded += _advance_iterator(shuffled_iterator, skip_to_checkpoint)
                    skip_to_checkpoint = 0  # done skipping
                # main inner loop over items
                for item in shuffled_iterator:
                    self._num_items_yielded += 1  # record how many items we have iterated over in this pass over the items
                    if (self._num_items_yielded-1) % self._num_instances == self._instance_rank:  # build-in islice facility
                        yield item
        self._iterator = _generate()

    def __next__(self):
        """
        Returns the next iterator.

        Args:
            self: (todo): write your description
        """
        return next(self._iterator)


class MultiplexIterator(CheckpointableIterator):
    """
    Multiplexes multiple input iterators.

    A control iterator is expected to yield a sequence of indices into an array of input iterators.
    The next item is selected from the input iterator whose index was read from the control iterator
    """
    def __init__(self, control_iterator: CheckpointableIterator, source_iterators: List[CheckpointableIterator]):
        """
        Initialize the control points.

        Args:
            self: (todo): write your description
            control_iterator: (todo): write your description
            source_iterators: (str): write your description
        """
        if any(not isinstance(it, CheckpointableIterator) for it in [control_iterator] + source_iterators):
            raise ValueError('control_iterator and source_iterators have to be CheckpointableIterators')
        self._control_iterator = control_iterator        # type: CheckpointableIterator
        self._source_iterators = list(source_iterators)  # type: List[CheckpointableIterator]
        self.setstate(None)
    
    def getstate(self) -> Dict:
        """
        Return the state of the source.

        Args:
            self: (todo): write your description
        """
        return {'control_iterator_state': self._control_iterator.getstate(),
                'source_iterator_states': [source_iterator.getstate() for source_iterator in self._source_iterators]}

    def setstate(self, checkpoint: Optional[Dict]):
        """
        Sets the state of the checkpoint.

        Args:
            self: (todo): write your description
            checkpoint: (bool): write your description
        """
        self._control_iterator.setstate(checkpoint['control_iterator_state'] if checkpoint else None)
        for i, source_iterator in enumerate(self._source_iterators):
            source_iterator.setstate(checkpoint['source_iterator_states'][i] if checkpoint else None)
        def _generate():
            """
            Generate an iterator that yields iterator.

            Args:
            """
            for index in self._control_iterator:
                item = next(self._source_iterators[index])
                yield item
        self._iterator = _generate()

    def __next__(self):
        """
        Returns the next iterator.

        Args:
            self: (todo): write your description
        """
        return next(self._iterator)


class SelectManyIterator(CheckpointableIterator):
    """
    Projects each element of a source sequence to a sequence and flattens the resulting sequences into one sequence.
    """
    def __init__(self, source_iterator: CheckpointableIterator, collection_selector: Optional[Callable[[Any], Iterator]]=None):
        """
        Args:
            source_iterator: iterator over the items to pass to collection_selector()
            collection_selector: user callback that maps an item into an Iterable, whose items will be yielded.
                                 The returned Iterator is used only once. Hence, it is also allowed to
                                 return self-iterables, such as iterators and generator expressions.
                                 If None is given, no callback is applied.
        """
        if not isinstance(source_iterator, CheckpointableIterator):
            raise ValueError('source_iterator has to be a CheckpointableIterator')
        self._source_iterator = source_iterator          # type: CheckpointableIterator
        self._collection_selector = collection_selector  # type: Optional[Callable[[Any], Iterator]]
        self.setstate(None)

    def getstate(self) -> Dict:
        """
        Return a dictionary of the source.

        Args:
            self: (todo): write your description
        """
        return {'source_state':            self._source_state,
                'flattened_items_yielded': self._flattened_items_yielded}

    def setstate(self, checkpoint: Optional[Dict]):
        """
        Set the state of this item.

        Args:
            self: (todo): write your description
            checkpoint: (bool): write your description
        """
        self._source_state            = checkpoint['source_state']            if checkpoint else None
        self._flattened_items_yielded = checkpoint['flattened_items_yielded'] if checkpoint else 0
        self._source_iterator.setstate(self._source_state)
        def _generate():
            """
            Generate items from a generator items.

            Args:
            """
            skip_to_checkpoint = self._flattened_items_yielded
            # main loop over source source_items
            for source_item in self._source_iterator:
                if self._collection_selector is not None:
                    data = iter(self._collection_selector(source_item))
                else:
                    data = iter(source_item)
                self._flattened_items_yielded = 0
                if skip_to_checkpoint:
                    #print("Skipping to index", skip_to_checkpoint, file=sys.stderr)
                    self._flattened_items_yielded += _advance_iterator(data, skip_to_checkpoint)
                    skip_to_checkpoint = 0
                # main loop over lines
                for item in data:
                    self._flattened_items_yielded += 1
                    yield item
                self._source_state = self._source_iterator.getstate()
        self._iterator = _generate()

    def __next__(self):
        """
        Returns the next iterator.

        Args:
            self: (todo): write your description
        """
        return next(self._iterator)


class BufferedShuffleIterator(CheckpointableIterator):
    """
    Shuffles given iterable using a limited buffer.
    """
    def __init__(self, source_iterator: CheckpointableIterator, buffer_size: int, seed: int=0):
        """
        Args:
            source_iterator: checkpointable iterator or restartable iterable over input items to shuffle
            buffer_size: size of the buffer in number of items used for shuffling
            seed: random seed used for shuffling (or None)
        """
        if not isinstance(source_iterator, CheckpointableIterator):
            raise ValueError('source_iterator has to be a CheckpointableIterator')
        self._source_iterator = source_iterator
        self._buffer_size = buffer_size
        self._seed = seed
        self.setstate(None)

    def getstate(self) -> Dict:
        """
        Get the state of the source.

        Args:
            self: (todo): write your description
        """
        return {'source_state': self._source_iterator.getstate(),
                'buffer':       copy.deepcopy(self._buffer),  # create deepcopy so that iterator cannot modify checkpoint after it was taken
                'random_state': self._random.getstate()}

    def setstate(self, checkpoint: Optional[Dict]):
        """
        Sets the state.

        Args:
            self: (todo): write your description
            checkpoint: (bool): write your description
        """
        if checkpoint:
            self._source_iterator.setstate(checkpoint['source_state'])
            self._buffer = copy.deepcopy(checkpoint['buffer'])  # create deepcopy so that iterator cannot modify checkpoint
            self._random.setstate(checkpoint['random_state'])
            # @TODO: Can we add a comment how the flush part is handled?
        else:
            self._source_iterator.setstate(None)
            self._buffer = [None for _ in range(self._buffer_size)]
            self._random = Random(self._seed)  # type: Random
        self._iterator = self._generate()

    def _generate(self) -> Iterator:
        """
        Generate a generator.

        Args:
            self: (todo): write your description
        """
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
        """
        Returns the next iterator.

        Args:
            self: (todo): write your description
        """
        return next(self._iterator)


class MapIterator(CheckpointableIterator):
    """
    Applies given tranform to each data item
    """
    def __init__(self, source_iterator: CheckpointableIterator, transform: Callable[[str],Any]):
        """
        Args:
            source_iterator: checkpointable iterator
            transform: function to be applied to each data item
        """
        if not isinstance(source_iterator, CheckpointableIterator):
            raise ValueError('source_iterator has to be a CheckpointableIterator')
        self._source_iterator = source_iterator
        self._transform = transform

    def getstate(self) -> Dict:
        """
        Return the state of the source.

        Args:
            self: (todo): write your description
        """
        return self._source_iterator.getstate()

    def setstate(self, checkpoint: Optional[Dict]):
        """
        Sets the checkpoint.

        Args:
            self: (todo): write your description
            checkpoint: (bool): write your description
        """
        self._source_iterator.setstate(checkpoint)

    def __next__(self):
        """
        Returns the next source.

        Args:
            self: (todo): write your description
        """
        return self._transform(next(self._source_iterator))


def ParallelMapIterator(source_iterator: CheckpointableIterator, transform: Callable[[str],Any], num_processes: int, num_items_per_process: int) -> CheckpointableIterator:
    """
    Applies given transform to each data item

    Behaves the same as MapIterator, but applies transform in parallel using multiple processes in a parallel map operation.

    Warning:
    The transform function has to be pickleable because it is sent across process boundaries.
    To achieve this, transform should be a top-level function.

    Args:
        source_iterator: checkpointable iterator
        transform: function to be applied to each data item, has to be pickleable, see above
        num_processes: number of processes to use for parallel map
        num_items_per_process: number of data items each process operates on
    """
    # divide stream of data items into batches
    batched_samples = FixedBatchIterator(source_iterator, num_processes * num_items_per_process)
    # create process pool and capture it in closure that performs parallel map
    p = python_multiprocessing.Pool(num_processes)
    def parallel_map_transform(buffer):
        """
        Parse a map.

        Args:
            buffer: (todo): write your description
        """
        return p.map(transform, buffer)
    # apply transform in parallel to data items in a batch
    batched_transformed_samples = MapIterator(batched_samples, parallel_map_transform)
    # unpack batches to go back to stream of (now transformed) data items
    transformed_samples = SelectManyIterator(batched_transformed_samples)
    return transformed_samples


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
        # TODO: Use all function?
        for source_iterator in source_iterators:
            if not isinstance(source_iterator, CheckpointableIterator):
                raise ValueError('all iterators in source_iterators have to be CheckpointableIterator')
        self._source_iterators = list(source_iterators)    # type: List[CheckpointableIterator]

    def getstate(self) -> Dict:
        """
        Return the state of the source.

        Args:
            self: (todo): write your description
        """
        return {'input_states': tuple(iterator.getstate() for iterator in self._source_iterators)}

    def setstate(self, checkpoint: Optional[Dict]):
        """
        Sets the checkpoint state.

        Args:
            self: (todo): write your description
            checkpoint: (bool): write your description
        """
        if checkpoint is None:
            for iterator in self._source_iterators:
                iterator.setstate(None)
        else:
            # TODO: Add check that both lists have the same length?
            for iterator, state in zip(self._source_iterators, checkpoint['input_states']):
                iterator.setstate(state)

    def __next__(self):
        """
        Returns the next iterator of elements.

        Args:
            self: (todo): write your description
        """
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

    E.g. [1, 2, 3, 4, 5, 6] with width = 3 will yield
    [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
    """
    def __init__(self, source_iterator: CheckpointableIterator, width: int):
        """
        Args:
            source_iterator: checkpointable input iterators
        """
        if not isinstance(source_iterator, CheckpointableIterator):
            raise ValueError('source_iterator has to be a CheckpointableIterator')
        self._source_iterator = source_iterator  # type: CheckpointableIterator
        self._width = width                      # type: int
        self.setstate(None)

    def getstate(self) -> Dict:
        """
        Return the state of the item.

        Args:
            self: (todo): write your description
        """
        return {'source_state': self._source_state,  # state for first item in FIFO
                'item_index':  self._item_index}   # index of next item to serve

    def setstate(self, checkpoint: Optional[Dict]):
        """
        Set the checkpoint state.

        Args:
            self: (todo): write your description
            checkpoint: (bool): write your description
        """
        self._source_state = checkpoint['source_state'] if checkpoint else None
        self._item_index   = checkpoint['item_index']   if checkpoint else 0
        self._source_iterator.setstate(self._source_state)
        self._iterator = self._generate()

    def _fifo_slice(self, i):  # returns a window into the FIFO beginning at i
        """
        Return slice slice of the slice.

        Args:
            self: (todo): write your description
            i: (todo): write your description
        """
        # @TODO: for efficiency, make this a slice view
        return tuple(self._fifo[i:i + self._width])

    def _generate(self) -> Iterator:
        """
        Generate a generator that yields a generator.

        Args:
            self: (todo): write your description
        """
        self._source_state = self._source_iterator.getstate()
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
            self._source_state = next_input_state  # this reflects now the first element in the FIFO
            self._item_index = 0

    def __next__(self):
        """
        Returns the next iterator.

        Args:
            self: (todo): write your description
        """
        return next(self._iterator)


# @TODO: research on whether this operation has a well-known name
class FixedBatchIterator(CheckpointableIterator):
    """
    Batches N consecutive items into a single item that is a list of these items.

    E.g. [1, 2, 3 4, 5, 6, 7, 8] with batch_size = 3 will yield
    [(1, 2, 3), (4, 5, 6), (7, 8)]
    """
    def __init__(self, source_iterator: CheckpointableIterator, batch_size: int):
        """
        Args:
            source_iterator: checkpointable input iterators
            batch_size: number of items per batch
        """
        if not isinstance(source_iterator, CheckpointableIterator):
            raise ValueError('source_iterator has to be a CheckpointableIterator')
        self._source_iterator = source_iterator  # type: CheckpointableIterator
        self._batch_size = batch_size            # type: int
        self.setstate(None)

    def getstate(self) -> Dict:
        """
        Get the state of the source.

        Args:
            self: (todo): write your description
        """
        return {'source_state': self._source_iterator.getstate()}  # state for first item in next batch

    def setstate(self, checkpoint: Optional[Dict]):
        """
        Set the checkpoint.

        Args:
            self: (todo): write your description
            checkpoint: (bool): write your description
        """
        self._source_state = checkpoint['source_state'] if checkpoint else None
        self._source_iterator.setstate(self._source_state)
        self._iterator = self._generate()

    def _generate(self) -> Iterator:
        """
        Generate a batch.

        Args:
            self: (todo): write your description
        """
        while True:
            batch = list(islice(self._source_iterator, self._batch_size))
            if not batch:
                break
            yield batch

    def __next__(self):
        """
        Returns the next iterator.

        Args:
            self: (todo): write your description
        """
        return next(self._iterator)


class RandomIterator(CheckpointableIterator):
    """
    Iterator to generate uniformly distributed random numbers in the interval [0,1).
    Very similar to Random.random(), except that random numbers are
    obtained via next().
    """
    def __init__(self, seed: int=0):
        """
        Args:
            seed: Random seed.
        """
        self._seed = seed
        self._random = Random(self._seed)  # type: Random

    def getstate(self) -> Dict:
        """
        Return a random state of this isstate.

        Args:
            self: (todo): write your description
        """
        return {'random_state': self._random.getstate()}

    def setstate(self, checkpoint: Optional[Dict]):
        """
        Sets the seed.

        Args:
            self: (todo): write your description
            checkpoint: (bool): write your description
        """
        if checkpoint is None:
            self._random.seed(self._seed)
        else:
            self._random.setstate(checkpoint['random_state'])
        
    def __next__(self):
        """
        Returns a random iterator.

        Args:
            self: (todo): write your description
        """
        return self._random.random()


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
        if not isinstance(source_iterator, CheckpointableIterator):
            raise ValueError('source_iterator has to be a CheckpointableIterator')
        self._source_iterator = source_iterator               # type: CheckpointableIterator
        self._step_function   = step_function                 # type: Callable[[Any,Any], Tuple[Any,Any]]
        # take deepcopy of initial state so that user cannot change initial state after iterator is created
        self._initial_state   = copy.deepcopy(initial_state)  # type: Any
        self.setstate(None)

    def getstate(self):
        """
        Returns a dictionary of the source.

        Args:
            self: (todo): write your description
        """
        # return deepcopy of recurrent state so that user cannot change recurrent state within a checkpoint after it was taken
        # by modifying the recurrent_state in place during the step_function
        return {'recurrent_state': copy.deepcopy(self._recurrent_state),
                'source_state':    self._source_iterator.getstate()}

    def setstate(self, checkpoint):
        """
        Sets a checkpoint.

        Args:
            self: (todo): write your description
            checkpoint: (bool): write your description
        """
        # take deepcopy of recurrent_state from checkpoint and initial state so that user cannot modify the checkpoint / the initial state
        # by modifying the recurrent_state in place during the step_function
        self._recurrent_state = copy.deepcopy(checkpoint['recurrent_state']) if checkpoint else copy.deepcopy(self._initial_state)
        self._source_iterator.setstate(checkpoint['source_state'] if checkpoint else None)
        def _generate():
            """
            Generate a list of the given the state.

            Args:
            """
            for item in self._source_iterator:
                # with all the deepcopies above, in-place modification of recurrent_state within the step_function is now ok
                self._recurrent_state, output = self._step_function(self._recurrent_state, item)
                yield output
        self._iterator = _generate()

    def __next__(self):
        """
        Returns the next iterator.

        Args:
            self: (todo): write your description
        """
        return next(self._iterator)


def SamplingRandomMapIterator(source_iterator: CheckpointableIterator, transform: Callable[[Random,Any],Any], seed: int=0):
    """
    An iterator that calls a transform function on each item, while also passing a checkpointed
    random generator.

    Args:
        source_iterator: checkpointable iterator to recur over
        step_function: user-supplied function with signature step_function(random, item) -> result_item
        seed: random seed
    """
    _random = Random(seed)
    def _step_function(state, item):
        """
        Perform a function.

        Args:
            state: (todo): write your description
            item: (todo): write your description
        """
        _random.setstate(state)
        output = transform(_random, item)
        return _random.getstate(), output
    return RecurrentIterator(source_iterator, _step_function, initial_state=_random.getstate())


def BlockwiseShuffleIterator(source_iterator: CheckpointableIterator, block_size: int, seed: int=0):
    """
    Shuffles a sequence of items by grouping consecutive items in blocks of fixed size, shuffling
    each block, and yielding the shuffled items of all blocks as a flat sequence.

    E.g. [1, 2, 3, 4, 5, 6, 7, 8] with block_size = 3 may yield [3, 1, 2, 4, 6, 5, 8, 7].

    Args:
        source_iterator: checkpointable iterator or restartable iterable over input items to shuffle
        block_size: size of the buffer in number of items used for shuffling
        seed: random seed used for shuffling (or None)
    """
    # This is implemented as a pipeline:
    #  - group N consecutive items together
    #  - shuffle them
    #  - flatten the result
    blocks = FixedBatchIterator(source_iterator, batch_size=block_size)
    def shuffle_block_fn(random: Random, block: List):
        """
        Shuffle a block of a block.

        Args:
            random: (todo): write your description
            block: (todo): write your description
        """
        random.shuffle(block)
        return block
    shuffled_blocks = SamplingRandomMapIterator(blocks, transform=shuffle_block_fn, seed=seed)
    samples = SelectManyIterator(shuffled_blocks, collection_selector=lambda shuffled_block: iter(shuffled_block))
    return samples


def PrefetchIterator(source_iterator: CheckpointableIterator, buffer_size: int, multiprocessing=None):
    """
    An iterator prefetching data into a buffer on a seperate process.

    Args:
        source_iterator: checkpointable iterator to recur over
        buffer_size: number of items to prefetch; this is the maximum number of items held in the prefetch queue
        multiprocessing: module to get `Queue` type from. Pass torch.multiprocessing here when items are Torch tensors for optimized data transfer.
    """
    if python_multiprocessing.get_start_method() != 'fork':
        print('WARNING: \
               PrefetchIterator is only supported on operating system that use fork to create new processes.\
               This excludes Windows.\
               A dummy iterator is inserted instead of a PrefetchIterator.\
               This also means that checkpoints of this iterator pipeline cannot be ported to a system that uses fork.')
        return source_iterator
    else:
        return _ForkPrefetchIterator(source_iterator, buffer_size, multiprocessing)


class _ForkPrefetchIterator(CheckpointableIterator):
    """
    Actual internal implementation of the prefetch iterator for systems that support creating processes through fork.

    Args:
        source_iterator: checkpointable iterator to recur over
        buffer_size: number of items to prefetch; this is the maximum number of items held in the prefetch queue
        multiprocessing_module: use this in place of Python's multiprocessing module, to allow for using torch.multiprocessing.Queue
    """
    def __init__(self, source_iterator: CheckpointableIterator, buffer_size: int, multiprocessing_module):
        """
        Initialize the module.

        Args:
            self: (todo): write your description
            source_iterator: (str): write your description
            buffer_size: (int): write your description
            multiprocessing_module: (todo): write your description
        """
        if not isinstance(source_iterator, CheckpointableIterator):
            raise ValueError('source_iterator has to be a CheckpointableIterator')
        self._source_iterator = source_iterator  # type:CheckpointableIterator
        self._buffer_size = buffer_size          # type: int
        self._QueueType = multiprocessing_module.Queue if multiprocessing_module else  \
                          python_multiprocessing.Queue
        self._prefetch_process = None            # type: Optional[python_multiprocessing.Process]
        self.setstate(None)

    def getstate(self) -> Dict:
        """
        Return the state of the state.

        Args:
            self: (todo): write your description
        """
        return {'source_state': self._source_state,
                'item_offset' : self._item_offset  }

    def setstate(self, checkpoint: Optional[Dict]):
        """
        Set the state.

        Args:
            self: (todo): write your description
            checkpoint: (bool): write your description
        """
        self._terminate_and_join_prefetch_process()  # kill current process if any

        self._source_state = checkpoint['source_state'] if checkpoint is not None else None
        self._item_offset  = checkpoint['item_offset' ] if checkpoint is not None else 0
        self._source_iterator.setstate(self._source_state)
        self._queue = self._QueueType(maxsize=self._buffer_size)
        _prefetch_process = python_multiprocessing.Process(target=self._prefetch_process_fn,
                                                           args=(self._source_iterator,
                                                                 self._item_offset,  # @TODO: why pass all these parameters? They are forked anyways. Seems a left-over from thread days.
                                                                 self._buffer_size,
                                                                 self._queue))
        _prefetch_process.start()  # this invokes fork()
        self._prefetch_process = _prefetch_process
        # make sure that in case of an unexpected shutdown, we still get rid of any active child process
        import atexit
        atexit.register(_ForkPrefetchIterator._join_process, self._prefetch_process)

    @staticmethod
    def _prefetch_process_fn(source, item_offset, buffer_size, queue):  # behavior of the prefetching process, only to be called from that process!        
        """
        Prefetch data from the queue.

        Args:
            source: (todo): write your description
            item_offset: (todo): write your description
            buffer_size: (int): write your description
            queue: (todo): write your description
        """
        _advance_iterator(source, item_offset)  # skip to checkpoint
        while True:
            try:
                item = next(source)
            except StopIteration:
                queue.put(StopIteration())
                # It seems Python Queue has a bug: if we return here, then the StopIteration message is never sent to the receiver.
                # So we just dead-loop, assuming that the process will be killed anyways when the consuming side destructs the prefetcher.
                import time
                while True:
                    time.sleep(1000)
                return  # we never actually get here
            if item_offset == buffer_size - 1:    # for efficiency, we send a new source state only at the END of each window of length _buffer_size
                source_state = source.getstate()  # this is the state for retrieving the NEXT element, i.e. the first element of the next buffer
                item_offset = 0
            else:
                source_state = None
                item_offset += 1
            msg = (item, source_state)
            queue.put(msg)

    def __next__(self):
        """
        Returns the next item in the queue.

        Args:
            self: (todo): write your description
        """
        if self._queue is None:  # iterator has already been exhausted
            raise StopIteration()
        msg = self._queue.get()
        if isinstance(msg, StopIteration):
            self._queue = None
            raise StopIteration()
        item, prefetch_source_state = msg  # for efficiency, the prefetch_source_state is only transmitted at the end of each window of length _buffer_size
        if prefetch_source_state is not None:
            assert self._item_offset == self._buffer_size - 1  # we expect a new source state at then END of each window of length _buffer_size
            self._source_state = prefetch_source_state
            self._item_offset = 0
        else:
            self._item_offset = self._item_offset + 1
            assert self._item_offset < self._buffer_size
        return item  # for debugging, its useful to return msg instead of item

    def __del__(self):  # note: this is often not called. If you really need it, gc.collect() will do the trick.
        """
        Terminate the process.

        Args:
            self: (todo): write your description
        """
        self._terminate_and_join_prefetch_process()

    def _terminate_and_join_prefetch_process(self):  # terminate the pre-fetch process if one is running
        """
        Terminate the prefetch.

        Args:
            self: (todo): write your description
        """
        if self._prefetch_process:
            _ForkPrefetchIterator._join_process(self._prefetch_process)
        self._prefetch_process = None

    @staticmethod
    def _join_process(p):  # called from setstate(), __del__(), and atexit handler
        """
        Terminate a process.

        Args:
            p: (todo): write your description
        """
        # We create prefetching processes with UNIX fork.
        # That means that we might end up with inactive copies
        # of prefetchers in the memory of prefetching processes.
        # These inactive copies can never create their
        # own prefetching processes, even if setstate is called.
        # All prefetching processes are exclusively created by
        # the main process, even if there are nested PrefetchIterators.
        # Hence, the main process should be the only one to terminate
        # and join prefetching processes.
        # The if-statement below guarantees that, even if __del__ is called
        # on a copy of a PrefetchIterator in another process
        if p._parent_pid != os.getpid():
            return
        if p.exitcode is not None:  # already joined: p.pid is invalid
            return
        # Note that we must terminate here instead of cleanly shutting down
        # the prefetching process, e.g. using synchronization primitives.
        # This is deliberate (and unfortunate).
        # The prefetching process might spend an arbitrary amount of time
        # in the preceeding iterators before it checks for potential termination messages.
        # This would hold up the entire pipeline due to the join below.
        # Hence, we terminate the process immediately.
        # In some cases, the process function already ran its course. In that case,
        # the terminate() call will have no effect.
        p.terminate()
        p.join()


class BucketedReadaheadBatchIterator(CheckpointableIterator):
    """
    Iterates over items from a checkpointable iterator and groups items of similar length into batches.

    The algorithm reads a head a certain number of lines (e.g. 10 million), sorts them by
    length, and them groups them into batches from start to end. The sort is stable, such
    that prior randomization is not undone (except for the length grouping). The batch size
    is dynamic, and determined by a user-provided callback.

    This is based on Marian NMT's BatchGenerator.
    """

    def __init__(self, source_iterator: CheckpointableIterator, read_ahead: int, key: Callable[[Any], Any], batch_size: Union[int,Callable[[Any], int]], shuffle: bool=True, seed: int=0):
        """
        Args:
            source_iterator: The data set that is read from. Typically this is an infinite source.
            read_ahead: Number of items to fetch ahead for grouping purposes.
            key: User-provided callback to define how data is sorted for purpose of batching.
            batch_size: Batch size in number of items. Either an integer or a callback to determine batch size for a given first batch item.
            shuffle: Pass False to not randomize the batches. (default: True)
            seed: Random seed for batch shuffling.
        """
        if not isinstance(source_iterator, CheckpointableIterator):
            raise ValueError('source_iterator has to be a CheckpointableIterator')
        # keep arguments
        self._key = key                # type: Callable[[Any], Any]
        self._batch_size = batch_size  # type: Union[int,Callable[[Any], int]]
        self._read_ahead = read_ahead  # type: int
        # initialize state
        self._seed = seed
        self._random = None            # type: Optional[Random]
        if shuffle:
            self._random = Random(self._seed)
        self._source_iterator = cast(CheckpointableIterator, iter(source_iterator))  # type: CheckpointableIterator
        self.setstate(None)

    def getstate(self):
        """
        Return a random state.

        Args:
            self: (todo): write your description
        """
        return {'source_state': self._source_state,
                'random_state': self._random_state,
                'num_served':   self._num_batches_yielded}

    def setstate(self, checkpoint: Optional[Dict]):
        """
        Sets up the next batch.

        Args:
            self: (todo): write your description
            checkpoint: (bool): write your description
        """
        self._source_state        = checkpoint['source_state'] if checkpoint else None  # type: Dict  # state of input before reading the current set of batches
        self._random_state        = checkpoint['random_state'] if checkpoint else None  # type: Any   # state of random generator at _source_state
        self._num_batches_yielded = checkpoint['num_served']   if checkpoint else 0     # type: int   # number of batches served from the current set of batches
        # checkpointing: restore to start of current set of batches
        self._source_iterator.setstate(self._source_state)
        if self._random:
            if self._random_state:
                self._random.setstate(self._random_state)
            else:
                self._random.seed(self._seed)
        self._source_exhausted = False  # type: bool  # set to True once we hit StopIteration on source
        def _generate():
            """
            Generate batches from the batches.

            Args:
            """
            skip_to_checkpoint = self._num_batches_yielded
            source_exhausted = False
            while not source_exhausted:
                # prefetch the readahead buffer
                self._source_state = self._source_iterator.getstate()
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
                self._num_batches_yielded = _advance_iterator(batches, skip_to_checkpoint)
                skip_to_checkpoint = 0
                # main loop over batches in current read-ahead section
                for batch in batches:
                    self._num_batches_yielded += 1
                    yield batch
        self._iterator = _generate()  # type: Iterator  # iterator into current set of batches

    def _create_batches(self, items: List[Any]) -> List[List[Any]]:  # helper to form batches from a list of items
        """
        Create batches from a list of items.

        Args:
            self: (todo): write your description
            items: (todo): write your description
        """
            # sort by length, longest first
            if self._key:
                items.sort(key=self._key, reverse=True)  # note: sort() is stable, so we won't undo any randomization besides the bucketing
            # group into batches
            cur_batch = None  # type: Optional[List[Any]]
            batches = []      # type: List[Any]
            for item in items:
                if not cur_batch:
                    batch_size = self._batch_size if isinstance(self._batch_size, int) else \
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
        """
        Returns the next iterator.

        Args:
            self: (todo): write your description
        """
        return next(self._iterator)
