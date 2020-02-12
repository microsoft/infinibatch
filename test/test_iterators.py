import gzip
import itertools
from random import Random
import os
import shutil
import tempfile
from typing import Iterable, Iterator, Any, Union
import unittest

from infinibatch.iterators import InfinitePermutationIterator, chunked_readlines_iterator, BufferedShuffleIterator, \
                                  NativeCheckpointableIterator, BucketedReadaheadBatchIterator, \
                                  MapIterator, ZipIterator, WindowedIterator, \
                                  RandomIterator, RecurrentIterator, SamplingRandomMapIterator
from infinibatch.datasets import chunked_dataset_iterator


class TestBase(unittest.TestCase):
    def setUp(self):
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

    def tearDown(self):
        shutil.rmtree(self.data_dir)
    
    def assertMultisetEqual(self, a, b):
        self.assertEqual(len(a), len(b))
        self.assertSetEqual(set(a), set(b))


class TestInfinitePermutationIterator(TestBase):
    def test_repeat_once(self):
        # This tests that two consecutive iterations through the test data yields differently ordered sequences.
        reader: Iterator[Any] = iter(InfinitePermutationIterator(self.flattened_test_data, 42))
        items0 = list(itertools.islice(reader, len(self.flattened_test_data)))
        items1 = list(itertools.islice(reader, len(self.flattened_test_data)))
        self.assertMultisetEqual(items0 + items1, self.flattened_test_data * 2)
        self.assertTrue(any(item0 != item1 for item0, item1 in zip(items0, items1)))

    def test_reiter_once(self):
        # This differs from test_repeat_once in that we use checkpoints.
        reader: Iterable[Any] = InfinitePermutationIterator(self.flattened_test_data, 42)
        checkpoint = reader.getstate()
        items0 = list(itertools.islice(reader, len(self.flattened_test_data)))
        reader.setstate(checkpoint)
        items1 = list(itertools.islice(reader, len(self.flattened_test_data)))
        self.assertMultisetEqual(items0 + items1, self.flattened_test_data * 2)
        self.assertSequenceEqual(items0, items1)

    def test_checkpointing(self):
        random = Random()
        for i in range(5):
            # random sequence lengths to for testing different configurations
            test_source_length        = random.randrange(5,25)
            test_first_output_length  = random.randrange(5,25)
            test_second_output_length = random.randrange(5,25)
            # source
            test_source = range(test_source_length)
            reader = InfinitePermutationIterator(test_source, seed=i)
            # fetch a first sequence
            _ = list(itertools.islice(reader, test_first_output_length))
            #print('items0', items0)
            # fetch a second sequence
            checkpoint = reader.getstate()
            items1a = list(itertools.islice(reader, test_second_output_length))
            #print('items1a', items1a)
            # fetch that second sequence again via checkpointing
            reader.setstate(checkpoint)
            items1b = list(itertools.islice(reader, test_second_output_length))
            #print('items1b', items1b)
            # must be the same
            self.assertTrue(items1a == items1b)


class TestNativeCheckpointableIterator(TestBase):
    def test(self):
        # go half-way through data and create checkpoint
        it = NativeCheckpointableIterator(list(range(100)))
        items = list(itertools.islice(it, 50))
        checkpoint = it.getstate()

        # resume from checkpoint
        it = NativeCheckpointableIterator(list(range(100)))
        it.setstate(checkpoint)
        items += list(it)

        self.assertListEqual(items, list(range(100)))

    def test_iterator_exception(self):
        self.assertRaises(ValueError, NativeCheckpointableIterator, iter(range(10)))


# @TODO: Move all tests of simple operators to the top, so that they can run first
class TestRecurrentIterator(TestBase):
    def test(self):
        n = 100
        seq = list(range(n))
        def step_function(prev_state, item):
            output = item + prev_state
            new_state = output
            return new_state, output
        it = RecurrentIterator(NativeCheckpointableIterator(seq), step_function, initial_state = 0)
        actual0 = list(itertools.islice(it, n * 3 // 10))
        checkpoint = it.getstate()
        actual1a = list(it)
        actual = actual0 + actual1a
        it.setstate(checkpoint)
        actual1b = list(it)
        expected = [0]
        for i in seq[1:]:
            expected.append(expected[-1] + i)
        self.assertListEqual(actual,   expected)  # basic operation
        self.assertListEqual(actual1a, actual1b)  # checkpointing


class TestSamplingRandomMapIterator(TestBase):
    def test(self):
        for seq in (range(100), [[1,2,3], [4,5], [6,7,8]]):
            n = len(seq)
            def _transform(random: Random, item: Union[int,Iterable]):
                if isinstance(item, int):  # first test case
                    return item + random.random()
                else:  # second test case
                    output = []
                    for i in item:
                        output.append(i + random.random())
                    return output
            it = SamplingRandomMapIterator(NativeCheckpointableIterator(seq), transform=_transform, seed=1)
            actual0 = list(itertools.islice(it, n * 3 // 10))
            checkpoint = it.getstate()
            actual1a = list(it)
            it.setstate(checkpoint)
            actual1b = list(it)
            self.assertListEqual(actual1a, actual1b)


class TestChunkedReadlinesIterator(TestBase):    
    def test(self):
        items = list(chunked_readlines_iterator(NativeCheckpointableIterator(self.chunk_file_paths)))
        self.assertListEqual(items, self.flattened_test_data)

    def test_different_line_endings(self):
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

        lf_data = list(chunked_readlines_iterator(NativeCheckpointableIterator([lf_file])))
        crlf_dat = list(chunked_readlines_iterator(NativeCheckpointableIterator([crlf_file])))
        self.assertListEqual(lf_data, crlf_dat)

        shutil.rmtree(lf_dir)
        shutil.rmtree(crlf_dir)

    def test_checkpointing(self):
        chunk_file_paths = (os.path.join(self.data_dir, subpath.name) for subpath in os.scandir(self.data_dir) if subpath.is_file() and subpath.name.endswith('.gz'))
        chunk_file_paths = InfinitePermutationIterator(chunk_file_paths, shuffle=False)  # using this as checkpointed cycle()
        random = Random(1)
        for _ in range(5):
            first_length = random.randrange(11,31)
            extra_length = random.randrange(11,33)
            dataset = chunked_readlines_iterator(chunk_file_paths)
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
        # work on copy of data in case data is modified by class
        items = list(BufferedShuffleIterator(NativeCheckpointableIterator(self.flattened_test_data.copy()), 971, 42))
        self.assertMultisetEqual(items, self.flattened_test_data)

    def test_shuffle_buffer_size_one(self):
        # work on copy of data in case data is modified by class
        items = list(BufferedShuffleIterator(NativeCheckpointableIterator(self.flattened_test_data.copy()), 1, 42))
        self.assertListEqual(items, self.flattened_test_data)


class TestMapIterator(TestBase):
    def test_transform(self):
        items = list(MapIterator(NativeCheckpointableIterator(range(100)), lambda x: x + 1))
        self.assertListEqual(items, list(range(1, 101)))


class TestZipIterator(TestBase):
    def test(self):
        n = 100
        seq1 = list(range(n))
        seq2 = list(i * i for i in range(n))
        it = ZipIterator(NativeCheckpointableIterator(seq1), NativeCheckpointableIterator(seq2))
        items0 = list(itertools.islice(it, n * 3 // 10))
        checkpoint = it.getstate()
        items1a = list(it)
        it.setstate(checkpoint)
        items1b = list(it)
        self.assertListEqual(items0 + items1a, list(zip(seq1, seq2)))  # basic function
        self.assertListEqual(items1a, items1b)                         # checkpointing


class TestWindowedIterator(TestBase):
    def test(self):
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
        n = 100
        it = RandomIterator(seed=1)
        _ = list(itertools.islice(it, n * 3 // 10))
        checkpoint = it.getstate()
        items1a = list(itertools.islice(it, n * 7 // 10))
        it.setstate(checkpoint)
        items1b = list(itertools.islice(it, n * 7 // 10))
        self.assertListEqual(items1a, items1b)


#class TestSamplingMapIterator(TestBase):
#    def test(self):
#        n = 100
#        seq = list(range(n))
#        it = SamplingMapIterator(NativeCheckpointableIterator(seq), sampling_transform=lambda rand_val, item: item + rand_val, seed=1)
#        _ = list(itertools.islice(it, n * 3 // 10))
#        checkpoint = it.getstate()
#        items1a = list(itertools.islice(it, n * 7 // 10))
#        it.setstate(checkpoint)
#        items1b = list(itertools.islice(it, n * 7 // 10))
#        self.assertListEqual(items1a, items1b)


class Testchunked_dataset_iterator(TestBase):
    def test_no_shuffle(self):
        items = list(itertools.islice(chunked_dataset_iterator(self.data_dir, shuffle=False), len(self.flattened_test_data)))
        self.assertListEqual(items, self.flattened_test_data)
    
    def test_other_files_present(self):
        with open(os.path.join(self.data_dir, 'i_do_not_belong_here.txt'), 'w') as f:
            f.write('really ...')
        items = list(itertools.islice(chunked_dataset_iterator(self.data_dir, shuffle=False), len(self.flattened_test_data)))
        self.assertListEqual(items, self.flattened_test_data)

    def test_transform(self):
        transform = lambda s: s + '!'
        modified_test_data = [transform(s) for s in self.flattened_test_data]
        items = list(itertools.islice(chunked_dataset_iterator(self.data_dir, shuffle=False, transform=transform), len(self.flattened_test_data)))
        self.assertListEqual(items, modified_test_data)

    def test_two_instances(self):
        dataset0 = chunked_dataset_iterator(self.data_dir, shuffle=False, num_instances=2, instance_rank=0)
        dataset1 = chunked_dataset_iterator(self.data_dir, shuffle=False, num_instances=2, instance_rank=1)
        items0 = list(itertools.islice(dataset0, len(self.test_data[0]) + len(self.test_data[2])))
        items1 = list(itertools.islice(dataset1, len(self.test_data[1]) + len(self.test_data[3])))
        self.assertMultisetEqual(set(items0 + items1), self.flattened_test_data)

    def test_checkpointing(self):
        random = Random(1)
        for i in range(5):
            first_length = random.randrange(11,21)
            extra_length = random.randrange(11,21)
            dataset = chunked_dataset_iterator(self.data_dir, shuffle=(i % 2 == 0), seed=i, num_instances=2, instance_rank=0)
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
            chunked_dataset_iterator(self.data_dir, shuffle=True, seed=1),
            read_ahead=100, seed=1,
            key=lambda line: len(line),
            batch_size=lambda line: batch_labels // (1+len(line)))
        batches1 = list(itertools.islice(bg, num_batches))
        # verify determinism
        bg = BucketedReadaheadBatchIterator(
            chunked_dataset_iterator(self.data_dir, shuffle=True, seed=1),
            read_ahead=100, seed=1,
            key=lambda line: len(line),
            batch_size=lambda line: batch_labels // (1+len(line)))
        batches2 = list(itertools.islice(bg, num_batches))
        print([(len(batch[0]), len(batch)) for batch in batches1])
        self.assertListEqual(batches1, batches2)

    def test_checkpointing(self):
        first_batches = 12
        extra_batches = 7
        batch_labels = 123
        bg = BucketedReadaheadBatchIterator(
            chunked_dataset_iterator(self.data_dir, shuffle=True, seed=1),
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
