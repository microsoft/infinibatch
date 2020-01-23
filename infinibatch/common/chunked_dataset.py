import gzip
import itertools
import os
from random import Random
from typing import Union, Iterable, Any


class _InfinitePermutationIterator:
    def __init__(self, iterable: Iterable[Any], seed: int):
        """
        Infinitely generates permutations of the items in the given iterable.

        Arguments:
        iterable -- input iterable
        seed -- random seed used for shuffling
        """
        self._items = list(iterable)
        self._random = Random(seed)

    def __iter__(self):
        while True:
            self._random.shuffle(self._items)
            for item in self._items:
                yield item


class _ChunkedDataIterator:
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


class _BufferedShuffleIterator:
    def __init__(self, iterable: Iterable[Any], buffer_size: int, seed: int):
        """
        Shuffles given iterable using a buffer.
        
        Arguments:
        iterable -- input iterable over items to shuffle
        buffer_size -- size of the buffer in number of items used for shuffling
        seed -- random seed used for shuffling
        """
        self._iterable = iterable
        self._buffer = [None for _ in range(buffer_size)]
        self._random = Random(seed)

    def __iter__(self):
        # shuffle data with a buffer:
        # this is similar to what the Fisher-Yates shuffle does,
        # but modified to run with a constant-size buffer
        # see https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
        # this was inspired by an algorithm implemented in Kaldi
        # see https://kaldi-asr.org/doc/nnet-shuffle-egs_8cc.html
        for item in self._iterable:
            index = self._random.randrange(0, len(self._buffer))
            if self._buffer[index] is not None:
                yield self._buffer[index]
            self._buffer[index] = item

        # flush buffer
        for item in self._buffer:
            if item is not None:
                yield item


# @TODO: Support non-zipped files.
# @TODO: Change default buffer size to a more reasonable value.
# @TODO: Support index files?
class ChunkedDataset:
    def __init__(self, paths: Union[str, Iterable[str]], shuffle: bool=True, buffer_size: int=2**20, transform=None, seed: int=None, num_instances: int=1, instance_rank: int=0):
        """
        Dataset reading data from gzipped chunks.

        This dataset infinitely repeats the data.

        Arguments:
        paths -- path, or list of paths, of directory containing dataset, i.e., a collection of .gz-files containing compressed text
        shuffle -- if true, the data is shuffled
        buffer_size -- size of the buffer in number of samples / data items used for shuffling
        transform -- transform to be applied to each data item
        seed -- random seed
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
        
        samples = _ChunkedDataIterator(chunks)
        if self._shuffle:
            # use different seed for BufferedShuffleGenerator
            buffered_shuffle_iterator_seed = self._seed
            if buffered_shuffle_iterator_seed is not None:
                buffered_shuffle_iterator_seed += 1
            samples = _BufferedShuffleIterator(samples, self._buffer_size, buffered_shuffle_iterator_seed)
        if self._transform is not None:
            samples = (self._transform(item) for item in samples)
        return iter(samples)
