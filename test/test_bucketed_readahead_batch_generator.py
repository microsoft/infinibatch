import gzip
import os
import shutil
import tempfile
import unittest

from infinibatch.bucketed_readahead_batch_generator import BucketedReadaheadBatchGenerator

# @TODO: Maybe move this into a test utils class? Or just use a real reader.
class FakeIterableDataset: # just to have something to work with
    _path: str            # pathname of the file
    #_f: Any #: io.TextIOWrapper  # the file we are reading from

    def __init__(self, path : str):
        self._path = path
        self._f = open(self._path, mode='rt', encoding="utf-8")

    def _rebuffer(self):
        self._f.seek(0)  #  --@BUGBUG: If it is not seekable (e.g. stdin), this must raise StopIteration

    def __next__(self):
        try:
            return next(self._f)
        except StopIteration:
            self._rebuffer()      # this simple class just repeats the input infinitely
            return next(self._f)  # should not fail

    def __iter__(self):
        return self

# test code --@TODO: put this into a proper command-line wrapper or a test project
ds = FakeIterableDataset("/home/fseide/factored-segmenter/src/FactoredSegmenter.cs")
batch_labels = 10000
bg = BucketedReadaheadBatchGenerator(ds, read_ahead=100, key=lambda line: len(line), batch_size_fn=lambda line: batch_labels // len(line))
for batch in bg:
    print(batch)
