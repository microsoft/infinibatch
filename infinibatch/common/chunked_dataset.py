import gzip
import itertools
import os
import random
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
        self.random = random.Random(seed)

    
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
    def __init__(self, iterable: Iterable, buffer_size: int, random: random.Random):
        """
        Shuffles given iterable using a buffer.
        
        Arguments:
        iterable -- input iterable
        buffer_size -- size of the buffer in number of samples used for shuffling
        random -- RNG used for shuffling
        """
        self.iterable = iterable
        self.buffer = [None for _ in range(buffer_size)]
        self.random = random


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
    def __init__(self, paths: Union[str, Iterable[str]], shuffle=True, buffer_size=1024, transform=None, seed: int=None, num_instances=1, instance_rank=0):
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
        if shuffle:
            self.random = random.Random()
            if seed:
                self.random.seed(seed)
        else:
            self.random = None
        self.num_instances = num_instances
        self.instance_rank = instance_rank


    def __iter__(self):
        if not self.shuffle:
            gen = itertools.cycle(self.chunk_file_paths)
        else:
            gen = InfinitePermutationIterator(self.chunk_file_paths, 42)

        if self.num_instances > 1:
            gen = itertools.islice(gen, self.instance_rank, None, self.num_instances)

        gen = ChunkedDataReader(gen)

        if self.shuffle:
            gen = BufferedShuffleIterator(gen, self.buffer_size, self.random)
        
        if self.transform is not None:
            gen = (self.transform(item) for item in gen)
        
        return gen.__iter__()
