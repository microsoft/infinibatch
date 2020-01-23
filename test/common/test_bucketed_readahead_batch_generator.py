import gzip
import os
import shutil
import tempfile
import unittest
import sys, inspect

from infinibatch.common.bucketed_readahead_batch_generator import BucketedReadaheadBatchIterator

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
            return next(self._f).rstrip()
        except StopIteration:
            self._rebuffer()               # this simple class just repeats the input infinitely
            return next(self._f).rstrip()  # should not fail

    def __iter__(self):
        return self

# e.g.
# python3.6 -m unittest discover -s ./test/ -p test_bucketed_readahead_batch_generator.py
# path = os.path.abspath(inspect.getfile(inspect.currentframe())) + "/../../infinibatch/common/chunked_dataset.py"  # (use one of our own source files as a source)
# print(path)
# ds = FakeIterableDataset("/home/fseide/factored-segmenter/src/FactoredSegmenter.cs")
# batch_labels = 500
# bg = BucketedReadaheadBatchIterator(ds, read_ahead=100, key=lambda line: len(line), batch_size=lambda line: batch_labels // (1+len(line)))
# i = 0
# for batch in bg:
#     i = i + 1
#     if (i > 20):
#         break
#     print(f"\n---- size {len(batch)} ---\n")
#     print("\n".join(batch))
