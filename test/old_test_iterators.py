import gzip
import itertools
from random import Random
import os
import shutil
import tempfile
from typing import Iterable, Iterator, Any, Union
import unittest
import pickle
import gc

from infinibatch.iterators import (
    create_source_iterator,
    ChunkedSourceIterator,
    InfinitePermutationSourceIterator,
    BufferedShuffleIterator,
    BlockwiseShuffleIterator,
    NativeCheckpointableIterator,
    BucketedReadaheadBatchIterator,
    MapIterator,
    ParallelMapIterator,
    ZipIterator,
    FixedBatchIterator,
    WindowedIterator,
    SelectManyIterator,
    RandomIterator,
    RecurrentIterator,
    SamplingRandomMapIterator,
    PrefetchIterator,
    MultiplexIterator,
)

# TODO:
#  - make sure that all iterators can be reset to a checkpoint even after they were exhausted
#  - make sure that all iterators can be reset to a checkpoint that was taken after the iterator was exhausted
#  - make sure that all iterators can be reset to a checkpoint at the beginning of the iteration
#  - refactor test cases that do not rely on TestCheckpointableIterator
#  - make sure every iterator is tested for correct checkpointing at the end of the iterator
#  - test whether iterators give same result when running from start twice (is seed reset correctly?)


class TestBase(unittest.TestCase):
    def setUp(self):
        self.test_data = [
            ["item number one", "item number two", "item number three", "item number four"],
            ["item number five"],
            [
                "item number six",
                "item number seven",
                "item number eight",
                "item number nine",
                "item number ten",
                "item number eleven",
            ],
            ["item number twelve", "item number thirteen", "item number fourteen",],
        ]

        self.flattened_test_data = []
        for chunk in self.test_data:
            for item in chunk:
                self.flattened_test_data.append(item)

        self.data_dir = tempfile.mkdtemp()
        self.chunk_file_paths = []
        for chunk_id, chunk in enumerate(self.test_data):
            file_name = os.path.join(self.data_dir, "chunk_" + str(chunk_id).zfill(10) + ".gz")
            self.chunk_file_paths.append(file_name)
            file_content = "\n".join(chunk)
            with gzip.open(file_name, "wt", encoding="utf-8") as f:
                f.write(file_content)

    @staticmethod
    def read_chunk(textfile_path: str) -> Iterator[str]:  # read_chunk_fn for chunked_dataset_iterator
        with gzip.open(textfile_path, "rt", encoding="utf-8") as f:
            return iter(f.read().splitlines())

    def tearDown(self):
        gc.collect()  # this will get the pre-fetch terminated in some tests, which otherwise may still want to read these files
        shutil.rmtree(self.data_dir)

    def assertMultisetEqual(self, a, b):
        self.assertEqual(len(a), len(b))
        self.assertSetEqual(set(a), set(b))


class TestBucketedReadaheadBatchIterator(TestBase):
    def txest_basic_functionality(self):
        num_batches = 13
        batch_labels = 75  # note: these settings imply a few iterations through the chunks
        # basic operation, should not crash
        bg = BucketedReadaheadBatchIterator(
            chunked_dataset_iterator(self.chunk_file_paths, self.read_chunk, shuffle=True, buffer_size=1000, seed=1),
            read_ahead=100,
            seed=1,
            key=lambda line: len(line),
            batch_size=lambda line: batch_labels // (1 + len(line)),
        )
        batches1 = list(itertools.islice(bg, num_batches))
        # verify determinism
        bg = BucketedReadaheadBatchIterator(
            chunked_dataset_iterator(self.chunk_file_paths, self.read_chunk, shuffle=True, buffer_size=1000, seed=1),
            read_ahead=100,
            seed=1,
            key=lambda line: len(line),
            batch_size=lambda line: batch_labels // (1 + len(line)),
        )
        batches2 = list(itertools.islice(bg, num_batches))
        print([(len(batch[0]), len(batch)) for batch in batches1])
        self.assertListEqual(batches1, batches2)

    def test_checkpointing(self):
        first_batches = 12
        extra_batches = 7
        batch_labels = 123
        bg = BucketedReadaheadBatchIterator(
            chunked_dataset_iterator(self.chunk_file_paths, self.read_chunk, shuffle=True, buffer_size=1000, seed=1),
            read_ahead=100,
            seed=1,
            key=lambda line: len(line),
            batch_size=lambda line: batch_labels // (1 + len(line)),
        )
        _ = list(itertools.islice(bg, first_batches))
        checkpoint = bg.getstate()
        batches1 = list(itertools.islice(bg, extra_batches))
        bg.setstate(checkpoint)
        batches2 = list(itertools.islice(bg, extra_batches))
        self.assertListEqual(batches1, batches2)


if __name__ == "__main__":
    unittest.main()
