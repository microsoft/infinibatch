from .iterators import InfinitePermutationIterator, SelectManyIterator, PrefetchIterator, BufferedShuffleIterator, BlockShuffleIterator, MapIterator
from .files_and_blobs import find_files, read_utf8_file
from typing import Union, Iterable, Iterator, Callable, Any, Optional, Dict
import os, sys

"""
This module contains common datasets, which are implemented as convenience functions that compose underlying Infinibatch iterators.
"""


def bump_seed(seed: Optional[int], step = 1):
    """
    Helper to bump a random seed if not None.
    """
    return None if seed is None else seed + 1


def chunked_dataset_iterator(chunk_refs: Iterable, read_chunk_fn: Callable[[Any], Iterator],
                             shuffle: bool=True, buffer_size: int=2**20,
                             transform: Callable[[Any],Any]=None,
                             prefetch_buffer_size: int=2**20,
                             seed: Optional[int]=None, num_instances: int=1, instance_rank: int=0,
                             use_block: bool=False):
    """
    Dataset reading data from gzipped chunks.

    This dataset infinitely repeats the data.

    Args:
        chunk_refs: references (such as path names) to chunk files
        read_chunk_fn: function(chunk_ref) -> Iterator to read a chunk's content into an iterator over its items, e.g. read a file and split into text lines
        shuffle: if true, the data is shuffled
        buffer_size: size of the buffer in number of samples / data items used for shuffling (default: 2**20)
        transform: transform to be applied to each data item (transform(Any) -> Any)
        prefetch_buffer_size: if not None, insert a prefetch iterator with this buffer size (default: 2**20)
        seed: random seed (or None)
        num_instances: number of instances of this dataset. Meant for use with multi-process data loading, e.g., in distributed training.
        instance_rank: rank of this instance of the dataset. Meant for use with multi-process data loading, e.g., in distributed training.
        use_block: temporary option to switch to BlockShuffleIterator (used for comparing against WindowsShuffleIterator). Will go away.
    """
    # set up the chunk reader
    randomized_chunk_refs  = InfinitePermutationIterator(chunk_refs, seed, shuffle=shuffle, num_instances=num_instances, instance_rank=instance_rank)
    # set up the item reader
    samples = SelectManyIterator(source_iterator=randomized_chunk_refs, collection_selector=read_chunk_fn)
    # wrap the I/O operation in a prefetch iterator
    # BUGBUG: This currently fails the test suite (although it seems to work)
    #if prefetch_buffer_size:
    #    samples = PrefetchIterator(samples, prefetch_buffer_size)
    # set up the item randomizer
    if shuffle:
        # use different seed for BufferedShuffleGenerator
        if (use_block):
            samples = BlockShuffleIterator(samples, buffer_size, bump_seed(seed, 1))
        else:
            samples = BufferedShuffleIterator(samples, buffer_size, bump_seed(seed, 1))
    # apply transform, if given
    if transform is not None:
        samples = MapIterator(samples, transform)
    # this is what we are serving out
    return samples
