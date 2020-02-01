import copy
from itertools import islice
from random import Random
from typing import Any, Generator, Iterator, List


class CheckpointIterator:
    def __next__(self):
        raise NotImplementedError()

    def __iter__(self):
        return self

    def checkpoint(self):
        raise NotImplementedError()

    def load_checkpoint(self, checkpoint):
        raise NotImplementedError()


class NativeIterator(CheckpointIterator):
    def __init__(self, iterator: Iterator[Any]):
        self._iterator = iterator
        self._consumed_items = 0

    def __next__(self):
        item = next(self._iterator)
        self._consumed_items += 1
        return item

    def checkpoint(self) -> List:
        return [self._consumed_items]

    def load_checkpoint(self, checkpoint):
        self._consumed_items = checkpoint[-1]
        for _ in range(self._consumed_items):
            next(self._iterator)


class BufferedShuffleIterator(CheckpointIterator):
    def __init__(self, input_iterator: CheckpointIterator, buffer_size: int, seed: int = 0):
        self._input_iterator = input_iterator
        self._buffer = [None for _ in range(buffer_size)]  # maybe do this lazily?
        self._random = Random(seed)
        self._generator = self._create_generator()

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

    def checkpoint(self):
        previous_checkpoint = self._input_iterator.checkpoint()
        local_checkpoint = [copy.deepcopy(self._buffer), self._random.getstate()]
        previous_checkpoint.append(local_checkpoint)
        return previous_checkpoint

    def load_checkpoint(self, checkpoint):
        local_checkpoint = checkpoint[-1]
        checkpoint = checkpoint[:-1]
        self._input_iterator.load_checkpoint(checkpoint)
        self._buffer = local_checkpoint[0]
        self._random.setstate(local_checkpoint[1])


if __name__ == '__main__':
    data_size = 10**5

    data = NativeIterator(iter(range(data_size)))
    shuffled_data = BufferedShuffleIterator(data, 100)
    not_checkpointed = list(shuffled_data)

    data = NativeIterator(iter(range(data_size)))
    shuffled_data = BufferedShuffleIterator(data, 100)
    checkpointed = list(islice(shuffled_data, 10000-10))

    checkpoint = shuffled_data.checkpoint()
    data = NativeIterator(iter(range(data_size)))
    shuffled_data = BufferedShuffleIterator(data, 100, 42)
    shuffled_data.load_checkpoint(checkpoint)
    checkpointed += list(shuffled_data)

    print(checkpointed == not_checkpointed)
