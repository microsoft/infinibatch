import gzip
import itertools
import os
from random import Random
from typing import Union, Iterable, Any


class InfinitePermutationIterator:
    def __init__(self, items: list, seed: int):
        """
        Infinitely generates permutations of the given items.

        Arguments:
        items -- list of items
        seed -- random seed used for shuffling
        """
        self.items = items.copy()
        self.random = Random(seed)

    def __iter__(self):
        while True:
            self.random.shuffle(self.items)
            for item in self.items:
                yield item


class ChunkedDataReader:
    def __init__(self, chunk_file_paths: Iterable[str]):
        """
        Reads data from chunks.
        
        Arguments:
        chunk_file_paths -- iterable of paths to chunk files
        """
        self.chunk_file_paths = chunk_file_paths
    
    def __iter__(self):
        for chunk_file_path in self.chunk_file_paths:
            with gzip.open(chunk_file_path, 'rt', encoding='utf-8') as f:
                data = f.read().splitlines()
            for item in data:
                yield item


class BufferedShuffleIterator:
    def __init__(self, iterable: Iterable, buffer_size: int, seed: int):
        """
        Shuffles given iterable using a buffer.
        
        Arguments:
        iterable -- input iterable
        buffer_size -- size of the buffer in number of samples used for shuffling
        seed -- random seed used for shuffling
        """
        self.iterable = iterable
        self.buffer = [None for _ in range(buffer_size)]
        self.random = Random(seed)

    def __iter__(self):
        # shuffle data with a buffer:
        # this is similar to what the Fisher-Yates shuffle does,
        # but modified to run with a constant-size buffer
        # see https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
        # this was inspired by an algorithm implemented in Kaldi
        # see https://kaldi-asr.org/doc/nnet-shuffle-egs_8cc.html
        for item in self.iterable:
            index = self.random.randrange(0, len(self.buffer))
            if self.buffer[index] is not None:
                yield self.buffer[index]
            self.buffer[index] = item

        # flush buffer
        for item in self.buffer:
            if item is not None:
                yield item


# @TODO: Support non-zipped files.
# @TODO: Change default buffer size to a more reasonable value.
# @TODO: Support index files?
class ChunkedDataset:
    def __init__(self, paths: Union[str, Iterable[str]], shuffle: bool=True, buffer_size: int=2**20, transform=None, seed: int=None, num_instances: int=1, instance_rank: int=0):
        """
        Dataset reading data from gzipped chunks.

        Arguments:
        paths -- path, or list of paths, of directory containing dataset, i.e., a collection of .gz-files containing compressed text
        shuffle -- if true, the data is shuffled
        buffer_size -- size of the buffer in number of samples / data items used for shuffling
        transform -- transform to be applied to each data item
        seed -- random seed
        """
        if isinstance(paths, str):  # handle single string
            paths = [paths]
        self.chunk_file_paths = []
        for path in paths:
            for subpath in os.scandir(path):
                if subpath.is_file() and subpath.name.endswith('.gz'):
                    self.chunk_file_paths.append(os.path.join(path, subpath.name))
        self.chunk_file_paths.sort()  # make sure file order is always the same, independent of OS
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.transform = transform
        self.seed = seed
        self.num_instances = num_instances
        self.instance_rank = instance_rank

    def __iter__(self):
        if not self.shuffle:
            chunks = itertools.cycle(self.chunk_file_paths)
        else:
            chunks = InfinitePermutationIterator(self.chunk_file_paths, self.seed)
        if self.num_instances > 1:
            chunks = itertools.islice(chunks, self.instance_rank, None, self.num_instances)
        
        samples = ChunkedDataReader(chunks)
        if self.shuffle:
            # use different seed for BufferedShuffleGenerator
            buffered_shuffle_iterator_seed = self.seed
            if buffered_shuffle_iterator_seed:
                buffered_shuffle_iterator_seed += 1
            samples = BufferedShuffleIterator(samples, self.buffer_size, buffered_shuffle_iterator_seed)
        if self.transform is not None:
            samples = (self.transform(item) for item in samples)
        return samples.__iter__()
