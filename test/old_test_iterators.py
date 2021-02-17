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
from infinibatch.datasets import chunked_dataset_iterator


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


class TestSourceIterator(unittest.TestCase):
    def test_exception(self):
        self.assertRaises(ValueError, create_source_iterator, [1], train=False, shuffle=True)


class Test_chunked_dataset_iterator(TestBase):
    def test_no_shuffle(self):
        items = list(
            itertools.islice(
                chunked_dataset_iterator(self.chunk_file_paths, self.read_chunk, shuffle=False, buffer_size=1000),
                len(self.flattened_test_data),
            )
        )
        self.assertListEqual(items, self.flattened_test_data)

    def test_other_files_present(self):
        with open(os.path.join(self.data_dir, "i_do_not_belong_here.txt"), "w") as f:
            f.write("really ...")
        items = list(
            itertools.islice(
                chunked_dataset_iterator(self.chunk_file_paths, self.read_chunk, shuffle=False, buffer_size=1000),
                len(self.flattened_test_data),
            )
        )
        self.assertListEqual(items, self.flattened_test_data)

    def test_transform(self):
        transform = lambda s: s + "!"
        modified_test_data = [transform(s) for s in self.flattened_test_data]
        items = list(
            itertools.islice(
                chunked_dataset_iterator(
                    self.chunk_file_paths, self.read_chunk, shuffle=False, buffer_size=1000, transform=transform
                ),
                len(self.flattened_test_data),
            )
        )
        self.assertListEqual(items, modified_test_data)

    def test_two_instances(self):
        dataset0 = chunked_dataset_iterator(
            self.chunk_file_paths, self.read_chunk, shuffle=False, buffer_size=1000, num_instances=2, instance_rank=0
        )
        dataset1 = chunked_dataset_iterator(
            self.chunk_file_paths, self.read_chunk, shuffle=False, buffer_size=1000, num_instances=2, instance_rank=1
        )
        items0 = list(itertools.islice(dataset0, len(self.test_data[0]) + len(self.test_data[2])))
        items1 = list(itertools.islice(dataset1, len(self.test_data[1]) + len(self.test_data[3])))
        self.assertMultisetEqual(set(items0 + items1), self.flattened_test_data)

    def test_checkpointing(self):
        random = Random(1)
        for use_windowed in (True, False):
            for i in range(2):
                first_length = random.randrange(11, 21)
                extra_length = random.randrange(11, 21)
                dataset = chunked_dataset_iterator(
                    self.chunk_file_paths,
                    self.read_chunk,
                    shuffle=(i % 2 == 0),
                    buffer_size=1000,
                    seed=i,
                    num_instances=2,
                    instance_rank=0,
                    use_windowed=use_windowed,
                )
                for _ in range(first_length):
                    next(dataset)
                checkpoint = dataset.getstate()
                items1 = list(itertools.islice(dataset, extra_length))
                dataset.setstate(checkpoint)
                items2 = list(itertools.islice(dataset, extra_length))
                self.assertListEqual(items1, items2)


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
