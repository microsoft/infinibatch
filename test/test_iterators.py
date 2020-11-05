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

from infinibatch.iterators import create_source_iterator, ChunkedSourceIterator, InfinitePermutationSourceIterator, BufferedShuffleIterator, BlockwiseShuffleIterator, \
                                  NativeCheckpointableIterator, BucketedReadaheadBatchIterator, \
                                  MapIterator, ParallelMapIterator, ZipIterator, FixedBatchIterator, WindowedIterator, SelectManyIterator, \
                                  RandomIterator, RecurrentIterator, SamplingRandomMapIterator, \
                                  PrefetchIterator, MultiplexIterator
from infinibatch.datasets import chunked_dataset_iterator


# TODO:
#  - make sure that all iterators can be reset to a checkpoint even after they were exhausted
#  - make sure that all iterators can be reset to a checkpoint that was taken after the iterator was exhausted
#  - make sure that all iterators can be reset to a checkpoint at the beginning of the iteration
#  - refactor test cases that do not rely on TestCheckpointableIterator
#  - make sure every iterator is tested for correct checkpointing at the end of the iterator
#  - test whether iterators give same result when running from start twice (is seed reset correctly?)

class TestCheckpointableIterator:
    """
    These are common test cases for CheckointableIterators
    
    Inherit from this class and set self.iterator and self.expected_result in the setUp function to use.
    """
    def test_basic(self):
        """
        Takes a list of the list of test results.

        Args:
            self: (todo): write your description
        """
        self.assertListEqual(list(self.iterator), self.expected_result)

    def test_checkpointing_from_start(self):
        """
        Test if all checkpointing updates.

        Args:
            self: (todo): write your description
        """
        for _ in range(len(self.expected_result)):
            next(self.iterator)
        self.iterator.setstate(None)
        self.assertListEqual(list(self.iterator), self.expected_result)

    def test_checkpointing_in_middle(self):
        """
        Test if all middleware in the list.

        Args:
            self: (todo): write your description
        """
        result = [next(self.iterator) for _ in range(len(self.expected_result) // 3)]
        self.iterator.setstate(self.iterator.getstate())
        result += [item for item in self.iterator]
        self.assertListEqual(result, self.expected_result)

    def test_checkpointing_at_end(self):
        """
        Test for at_checkpoint at each point.

        Args:
            self: (todo): write your description
        """
        for _ in range(len(self.expected_result)):
            next(self.iterator)
        self.iterator.setstate(self.iterator.getstate())
        self.assertRaises(StopIteration, self.iterator.__next__)


class TestBase(unittest.TestCase):
    def setUp(self):
        """
        Creates the test data file.

        Args:
            self: (todo): write your description
        """
        self.test_data = \
        [
            [
                'item number one',
                'item number two',
                'item number three',
                'item number four'
            ],
            [
                'item number five'
            ],
            [
                'item number six',
                'item number seven',
                'item number eight',
                'item number nine',
                'item number ten',
                'item number eleven'
            ],
            [
                'item number twelve',
                'item number thirteen',
                'item number fourteen',
            ],
        ]

        self.flattened_test_data = []
        for chunk in self.test_data:
            for item in chunk:
                self.flattened_test_data.append(item)

        self.data_dir = tempfile.mkdtemp()
        self.chunk_file_paths = []
        for chunk_id, chunk in enumerate(self.test_data):
            file_name = os.path.join(self.data_dir, 'chunk_' + str(chunk_id).zfill(10) + '.gz')
            self.chunk_file_paths.append(file_name)
            file_content = '\n'.join(chunk)
            with gzip.open(file_name, 'wt', encoding='utf-8') as f:
                f.write(file_content)

    @staticmethod
    def read_chunk(textfile_path: str) -> Iterator[str]:   # read_chunk_fn for chunked_dataset_iterator
        """
        Reads a file.

        Args:
            textfile_path: (str): write your description
        """
        with gzip.open(textfile_path, 'rt', encoding='utf-8') as f:
            return iter(f.read().splitlines())

    def tearDown(self):
        """
        Tear down directory.

        Args:
            self: (todo): write your description
        """
        gc.collect()  # this will get the pre-fetch terminated in some tests, which otherwise may still want to read these files
        shutil.rmtree(self.data_dir)
    
    def assertMultisetEqual(self, a, b):
        """
        Asserts that two scalarrays.

        Args:
            self: (todo): write your description
            a: (todo): write your description
            b: (todo): write your description
        """
        self.assertEqual(len(a), len(b))
        self.assertSetEqual(set(a), set(b))


class TestMultiplexIterator(unittest.TestCase, TestCheckpointableIterator):
    def setUp(self):
        """
        Sets the two sequences of the dfa.

        Args:
            self: (todo): write your description
        """
        index_seq = [0, 2, 1, 2, 0, 1, 1]
        data_seqs = [[0.0, 0.1, 0.2, 0.3],
                     [1.0, 1.1, 1.2, 1.3],
                     [2.0, 2.1, 2.2, 2.3]]
        self.expected_result = [0.0, 2.0, 1.0, 2.1, 0.1, 1.1, 1.2]
        self.iterator = MultiplexIterator(NativeCheckpointableIterator(index_seq), [NativeCheckpointableIterator(ds) for ds in data_seqs])


class TestSourceIterator(unittest.TestCase):
    def test_exception(self):
        """
        Assigns the test dataset.

        Args:
            self: (todo): write your description
        """
        self.assertRaises(ValueError, create_source_iterator, [1], train=False, shuffle=True)


class TestChunkedSourceIterator(unittest.TestCase, TestCheckpointableIterator):
    def setUp(self):
        """
        Sets the result of this query.

        Args:
            self: (todo): write your description
        """
        self.expected_result = list(range(53))
        self.iterator = ChunkedSourceIterator(self.expected_result)

    def test_multiple_instance(self):
        """
        Test for multiple instances.

        Args:
            self: (todo): write your description
        """
        for num_instances in range(2, 17):
            items = []
            for rank in range(num_instances):
                iterator = ChunkedSourceIterator(self.expected_result, num_instances=num_instances, instance_rank=rank)
                items.extend(list(iterator))
            self.assertListEqual(items, self.expected_result)


class TestInfinitePermutationSourceIterator(TestBase):
    def test_repeat_once(self):
        """
        Test for iterable.

        Args:
            self: (todo): write your description
        """
        # This tests that two consecutive iterations through the test data yields differently ordered sequences.
        reader = iter(InfinitePermutationSourceIterator(self.flattened_test_data, 42))
        items0 = list(itertools.islice(reader, len(self.flattened_test_data)))
        items1 = list(itertools.islice(reader, len(self.flattened_test_data)))
        self.assertMultisetEqual(items0 + items1, self.flattened_test_data * 2)
        self.assertTrue(any(item0 != item1 for item0, item1 in zip(items0, items1)))

    def test_reiter_once(self):
        """
        Test whether the test.

        Args:
            self: (todo): write your description
        """
        # This differs from test_repeat_once in that we use checkpoints.
        reader = InfinitePermutationSourceIterator(self.flattened_test_data, 42)
        checkpoint = reader.getstate()
        items0 = list(itertools.islice(reader, len(self.flattened_test_data)))
        reader.setstate(checkpoint)
        items1 = list(itertools.islice(reader, len(self.flattened_test_data)))
        self.assertMultisetEqual(items0 + items1, self.flattened_test_data * 2)
        self.assertSequenceEqual(items0, items1)

    def test_checkpointing(self):
        """
        : parameter test : classifier.

        Args:
            self: (todo): write your description
        """
        random = Random()
        for i in range(5):
            # random sequence lengths to for testing different configurations
            test_source_length        = random.randrange(5,25)
            test_first_output_length  = random.randrange(5,25)
            test_second_output_length = random.randrange(5,25)
            # source
            test_source = list(range(test_source_length))
            reader = InfinitePermutationSourceIterator(test_source, seed=i)
            # fetch a first sequence
            _ = list(itertools.islice(reader, test_first_output_length))
            # fetch a second sequence
            checkpoint = reader.getstate()
            items1a = list(itertools.islice(reader, test_second_output_length))
            # fetch that second sequence again via checkpointing
            reader.setstate(checkpoint)
            items1b = list(itertools.islice(reader, test_second_output_length))
            # and again with serialized checkpoint
            as_json = pickle.dumps(checkpoint)
            checkpoint2 = pickle.loads(as_json)
            reader.setstate(checkpoint2)
            items1c = list(itertools.islice(reader, test_second_output_length))
            # must be the same
            self.assertTrue(items1a == items1b)
            self.assertTrue(items1a == items1c)


class TestNativeCheckpointableIterator(unittest.TestCase, TestCheckpointableIterator):
    def setUp(self):
        """
        Sets the result of this query.

        Args:
            self: (todo): write your description
        """
        self.expected_result = list(range(53))
        self.iterator = NativeCheckpointableIterator(self.expected_result)

    def test_iterator_exception(self):
        """
        Iterate the exception.

        Args:
            self: (todo): write your description
        """
        self.assertRaises(ValueError, NativeCheckpointableIterator, iter(range(10)))


class TestRecurrentIterator(unittest.TestCase, TestCheckpointableIterator):
    def setUp(self):
        """
        Set the state of each state.

        Args:
            self: (todo): write your description
        """
        data = list(range(53))

        self.expected_result = [0]
        for i in data[1:]:
            self.expected_result.append(self.expected_result[-1] + i)

        def step_function(prev_state, item):
            """
            Return the next state function.

            Args:
                prev_state: (todo): write your description
                item: (todo): write your description
            """
            output = item + prev_state
            new_state = output
            return new_state, output
        self.iterator = RecurrentIterator(NativeCheckpointableIterator(data), step_function, initial_state = 0)


class TestSamplingRandomMapIterator(unittest.TestCase, TestCheckpointableIterator):
    def setUp(self):
        """
        Sets the seed of each item. : attr : parameter. set.

        Args:
            self: (todo): write your description
        """
        data = list(range(53))
        def transform(random: Random, item: int):
            """
            Return a random item.

            Args:
                random: (array): write your description
                item: (array): write your description
            """
            return item + random.random()

        seed = 1
        random = Random()
        random.seed(seed)
        self.expected_result = [n + random.random() for n in data]

        self.iterator = SamplingRandomMapIterator(NativeCheckpointableIterator(data), transform=transform, seed=seed)


class TestFixedBatchIterator(unittest.TestCase, TestCheckpointableIterator):
    def setUp(self):
        """
        Adds a list of the batchpoint to the same.

        Args:
            self: (todo): write your description
        """
        data = list(range(5))

        batch_size = 3
        self.expected_result = [data[0:3], data[3:]]

        self.iterator = FixedBatchIterator(NativeCheckpointableIterator(data), batch_size=batch_size)


class TestSelectManyIterator(TestBase):
    # in this test, SelectManyIterator is used to read chunk files
    @staticmethod
    def _select_many_from_chunks(chunk_file_paths):
        """
        Select multiple chunks from a collection.

        Args:
            chunk_file_paths: (str): write your description
        """
        return SelectManyIterator(source_iterator=chunk_file_paths, collection_selector=TestBase.read_chunk)

    def test(self):
        """
        Flattens the list of - like this list of files

        Args:
            self: (todo): write your description
        """
        items = list(self._select_many_from_chunks(NativeCheckpointableIterator(self.chunk_file_paths)))
        self.assertListEqual(items, self.flattened_test_data)

    def test_no_selector(self):
        """
        Test if the selector matches the selector.

        Args:
            self: (todo): write your description
        """
        data = list(range(100))
        sublists = [data[:10], data[10:42], data[42: 87], data[87:]]
        result = list(SelectManyIterator(NativeCheckpointableIterator(sublists)))
        self.assertListEqual(result, data)

    def test_different_line_endings(self):
        """
        Writes endings.

        Args:
            self: (todo): write your description
        """
        # write data in binary mode with LF line endings
        lf_dir = tempfile.mkdtemp()
        lf_file = os.path.join(lf_dir, 'test.gz')
        with gzip.open(lf_file, 'w') as f:
            f.write('\n'.join(self.flattened_test_data).encode('utf-8'))

        # write data in binary mode with CRLF line endings
        crlf_dir = tempfile.mkdtemp()
        crlf_file = os.path.join(crlf_dir, 'test.gz')
        with gzip.open(crlf_file, 'w') as f:
            f.write('\r\n'.join(self.flattened_test_data).encode('utf-8'))

        lf_data = list(self._select_many_from_chunks(NativeCheckpointableIterator([lf_file])))
        crlf_dat = list(self._select_many_from_chunks(NativeCheckpointableIterator([crlf_file])))
        self.assertListEqual(lf_data, crlf_dat)

        shutil.rmtree(lf_dir)
        shutil.rmtree(crlf_dir)

    def test_checkpointing(self):
        """
        Test for test dataset.

        Args:
            self: (todo): write your description
        """
        chunk_file_paths = [os.path.join(self.data_dir, subpath.name) for subpath in os.scandir(self.data_dir) if subpath.is_file() and subpath.name.endswith('.gz')]
        chunk_file_paths = InfinitePermutationSourceIterator(chunk_file_paths, shuffle=False)  # using this as checkpointed cycle()
        random = Random(1)
        for _ in range(5):
            first_length = random.randrange(11,31)
            extra_length = random.randrange(11,33)
            dataset = self._select_many_from_chunks(chunk_file_paths)
            for _ in range(first_length):
                next(dataset)
            checkpoint = dataset.getstate()
            items0 = list(itertools.islice(dataset, extra_length))
            #print(len(items0))
            dataset.setstate(checkpoint)
            items1 = list(itertools.islice(dataset, extra_length))
            #print(len(items1))
            self.assertListEqual(items0, items1)


class TestBufferedShuffleIterator(TestBase):
    def test_shuffle(self):
        """
        Flattens the test set.

        Args:
            self: (todo): write your description
        """
        # work on copy of data in case data is modified by class
        items = list(BufferedShuffleIterator(NativeCheckpointableIterator(self.flattened_test_data.copy()), 971, 42))
        self.assertMultisetEqual(items, self.flattened_test_data)

    def test_shuffle_buffer_size_one(self):
        """
        Shuffle the test buffer size.

        Args:
            self: (todo): write your description
        """
        # work on copy of data in case data is modified by class
        items = list(BufferedShuffleIterator(NativeCheckpointableIterator(self.flattened_test_data.copy()), 1, 42))
        self.assertListEqual(items, self.flattened_test_data)


# note: this is also tested in more depth in Test_chunked_dataset_iterator()
class TestBlockwiseShuffleIterator(TestBase):
    def test_shuffle(self):
        """
        Shuffle the test set.

        Args:
            self: (todo): write your description
        """
        # work on copy of data in case data is modified by class
        items = list(BlockwiseShuffleIterator(NativeCheckpointableIterator(self.flattened_test_data.copy()), 971, 42))
        self.assertMultisetEqual(items, self.flattened_test_data)

    def test_shuffle_buffer_size_one(self):
        """
        Shuffle the test buffer size of the test buffer.

        Args:
            self: (todo): write your description
        """
        # work on copy of data in case data is modified by class
        items = list(BlockwiseShuffleIterator(NativeCheckpointableIterator(self.flattened_test_data.copy()), 1, 42))
        self.assertListEqual(items, self.flattened_test_data)


def map_fun(n):
    """
    Map a function to a list of integers.

    Args:
        n: (todo): write your description
    """
    return n + 1


class TestMapIterator(unittest.TestCase, TestCheckpointableIterator):
    def setUp(self):
        """
        Sets the list of the mapable map.

        Args:
            self: (todo): write your description
        """
        data = list(range(53))
        self.expected_result = [map_fun(n) for n in data]
        self.iterator = MapIterator(NativeCheckpointableIterator(data), map_fun)


class TestParallelMapIterator(unittest.TestCase, TestCheckpointableIterator):
    def setUp(self):
        """
        Sets the mapable function.

        Args:
            self: (todo): write your description
        """
        data = list(range(53))
        self.expected_result = [map_fun(n) for n in data]
        self.iterator = ParallelMapIterator(NativeCheckpointableIterator(data), map_fun, 5, 7)


class TestZipIterator(unittest.TestCase, TestCheckpointableIterator):
    def setUp(self):
        """
        Sets the list of the list of checkpointable points.

        Args:
            self: (todo): write your description
        """
        data1 = list(range(53))
        data2 = [n * n for n in data1]
        self.expected_result = list(zip(data1, data2))
        self.iterator = ZipIterator(NativeCheckpointableIterator(data1), NativeCheckpointableIterator(data2))


class TestWindowedIterator(TestBase):
    def test(self):
        """
        Test for test function.

        Args:
            self: (todo): write your description
        """
        for n in [0, 2, 3, 8, 9, 10, 11, 12]:  # cover various boundary conditions
            seq = list(range(n))
            it = WindowedIterator(NativeCheckpointableIterator(seq), 3)
            actual0 = list(itertools.islice(it, n * 3 // 10))
            checkpoint = it.getstate()
            actual1a = list(it)
            it.setstate(checkpoint)
            actual1b = list(it)
            actual = actual0 + actual1a
            expected = list(zip(seq, itertools.islice(seq, 1, None), itertools.islice(seq, 2, None)))
            self.assertListEqual(actual, expected)    # basic operation
            self.assertListEqual(actual1a, actual1b)  # checkpointing


class TestRandomIterator(TestBase):
    def test(self):
        """
        Test if all items in the state.

        Args:
            self: (todo): write your description
        """
        n = 100
        it = RandomIterator(seed=1)
        _ = list(itertools.islice(it, n * 3 // 10))
        checkpoint = it.getstate()
        items1a = list(itertools.islice(it, n * 7 // 10))
        it.setstate(checkpoint)
        items1b = list(itertools.islice(it, n * 7 // 10))
        self.assertListEqual(items1a, items1b)


class TestPrefetchIterator(unittest.TestCase, TestCheckpointableIterator):
    def setUp(self):
        """
        Sets the result of this query.

        Args:
            self: (todo): write your description
        """
        self.expected_result = list(range(53))
        source_iterator = NativeCheckpointableIterator(self.expected_result)
        self.iterator = PrefetchIterator(source_iterator, buffer_size=13)


class Test_chunked_dataset_iterator(TestBase):
    def test_no_shuffle(self):
        """
        Test if the test dataset files.

        Args:
            self: (todo): write your description
        """
        items = list(itertools.islice(chunked_dataset_iterator(self.chunk_file_paths, self.read_chunk, shuffle=False, buffer_size=1000), len(self.flattened_test_data)))
        self.assertListEqual(items, self.flattened_test_data)
    
    def test_other_files_present(self):
        """
        Check if there are present.

        Args:
            self: (todo): write your description
        """
        with open(os.path.join(self.data_dir, 'i_do_not_belong_here.txt'), 'w') as f:
            f.write('really ...')
        items = list(itertools.islice(chunked_dataset_iterator(self.chunk_file_paths, self.read_chunk, shuffle=False, buffer_size=1000), len(self.flattened_test_data)))
        self.assertListEqual(items, self.flattened_test_data)

    def test_transform(self):
        """
        Flattens the dataset.

        Args:
            self: (todo): write your description
        """
        transform = lambda s: s + '!'
        modified_test_data = [transform(s) for s in self.flattened_test_data]
        items = list(itertools.islice(chunked_dataset_iterator(self.chunk_file_paths, self.read_chunk, shuffle=False, buffer_size=1000, transform=transform), len(self.flattened_test_data)))
        self.assertListEqual(items, modified_test_data)

    def test_two_instances(self):
        """
        Test for two files.

        Args:
            self: (todo): write your description
        """
        dataset0 = chunked_dataset_iterator(self.chunk_file_paths, self.read_chunk, shuffle=False, buffer_size=1000, num_instances=2, instance_rank=0)
        dataset1 = chunked_dataset_iterator(self.chunk_file_paths, self.read_chunk, shuffle=False, buffer_size=1000, num_instances=2, instance_rank=1)
        items0 = list(itertools.islice(dataset0, len(self.test_data[0]) + len(self.test_data[2])))
        items1 = list(itertools.islice(dataset1, len(self.test_data[1]) + len(self.test_data[3])))
        self.assertMultisetEqual(set(items0 + items1), self.flattened_test_data)

    def test_checkpointing(self):
        """
        Generate the dataset.

        Args:
            self: (todo): write your description
        """
        random = Random(1)
        for use_windowed in (True, False):
            for i in range(2):
                first_length = random.randrange(11,21)
                extra_length = random.randrange(11,21)
                dataset = chunked_dataset_iterator(self.chunk_file_paths, self.read_chunk, shuffle=(i % 2 == 0), buffer_size=1000, seed=i, num_instances=2, instance_rank=0, use_windowed=use_windowed)
                for _ in range(first_length):
                    next(dataset)
                checkpoint = dataset.getstate()
                items1 = list(itertools.islice(dataset, extra_length))
                dataset.setstate(checkpoint)
                items2 = list(itertools.islice(dataset, extra_length))
                self.assertListEqual(items1, items2)


class TestBucketedReadaheadBatchIterator(TestBase):
    def txest_basic_functionality(self):
        """
        Txest batchality batch batch batch.

        Args:
            self: (todo): write your description
        """
        num_batches = 13
        batch_labels = 75  # note: these settings imply a few iterations through the chunks
        # basic operation, should not crash
        bg = BucketedReadaheadBatchIterator(
            chunked_dataset_iterator(self.chunk_file_paths, self.read_chunk, shuffle=True, buffer_size=1000, seed=1),
            read_ahead=100, seed=1,
            key=lambda line: len(line),
            batch_size=lambda line: batch_labels // (1+len(line)))
        batches1 = list(itertools.islice(bg, num_batches))
        # verify determinism
        bg = BucketedReadaheadBatchIterator(
            chunked_dataset_iterator(self.chunk_file_paths, self.read_chunk, shuffle=True, buffer_size=1000, seed=1),
            read_ahead=100, seed=1,
            key=lambda line: len(line),
            batch_size=lambda line: batch_labels // (1+len(line)))
        batches2 = list(itertools.islice(bg, num_batches))
        print([(len(batch[0]), len(batch)) for batch in batches1])
        self.assertListEqual(batches1, batches2)

    def test_checkpointing(self):
        """
        Test if the next checkpoint.

        Args:
            self: (todo): write your description
        """
        first_batches = 12
        extra_batches = 7
        batch_labels = 123
        bg = BucketedReadaheadBatchIterator(
            chunked_dataset_iterator(self.chunk_file_paths, self.read_chunk, shuffle=True, buffer_size=1000, seed=1),
            read_ahead=100, seed=1,
            key=lambda line: len(line),
            batch_size=lambda line: batch_labels // (1+len(line)))
        _ = list(itertools.islice(bg, first_batches))
        checkpoint = bg.getstate()
        batches1 = list(itertools.islice(bg, extra_batches))
        bg.setstate(checkpoint)
        batches2 = list(itertools.islice(bg, extra_batches))
        self.assertListEqual(batches1, batches2)


if __name__ == '__main__':
    unittest.main()
