import gzip
import os
from random import Random
from typing import Union, Iterable, Any

def chunked_data_generator(chunk_file_paths, random: Random):
    """
    Read and yield data from chunks.
    
    Arguments:
    chunk_file_paths -- list of paths to chunk files
    shuffle_chunks -- if true, the chunks are read in shuffled orders
    """
    chunk_file_paths = chunk_file_paths.copy()

    if random:
        random.shuffle(chunk_file_paths)
        
    for chunk_file_path in chunk_file_paths:
        with gzip.open(chunk_file_path, 'rt', encoding='utf-8') as f:
            data = f.read().splitlines()

        for item in data:
            yield item


def buffered_shuffle_generator(data, buffer_size, random: Random):
    """
    Shuffle and yield given data using a buffer.
    
    Arguments:
    data -- iterable containing data
    buffer_size -- size of the buffer in number of samples / data items used for shuffling
    """
    if buffer_size <= 0:
        raise ValueError('The buffer size must be positive.')

    buffer = [None for _ in range(buffer_size)]

    # shuffle data with a buffer:
    # this is similar to what the Fisher-Yates shuffle does,
    # but modified to run with a constant-size buffer
    # see https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
    # this was inspired by an algorithm implemented in Kaldi
    # see https://kaldi-asr.org/doc/nnet-shuffle-egs_8cc.html
    for item in data:
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
class ChunkedDataset:
    def __init__(self, paths: Union[str, Iterable[str]], shuffle=True, buffer_size=1024, transform=None, seed: int=None):
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
        self.random = Random()
        if seed is not None:
            self.random.seed(seed)


    def __iter__(self):
        gen = chunked_data_generator(self.chunk_file_paths, self.random)
        if self.random:
            gen = buffered_shuffle_generator(gen, self.buffer_size, self.random)
        if self.transform is not None:
            gen = (self.transform(item) for item in gen)
        return gen
