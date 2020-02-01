import gzip
import itertools
import os
import shutil
import tempfile
import unittest
from typing import Iterable, Iterator, Any

from infinibatch.common.chunked_dataset import IterableChunkedDataset, _InfinitePermutationIterator, _IterableChunkedData, _IterableBufferedShuffler


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
        reader: Iterator[Any] = iter(_InfinitePermutationIterator(self.flattened_test_data, 42))
        items0 = list(itertools.islice(reader, len(self.flattened_test_data)))
        items1 = list(itertools.islice(reader, len(self.flattened_test_data)))
        self.assertMultisetEqual(items0 + items1, self.flattened_test_data * 2)
        self.assertTrue(any(item0 != item1 for item0, item1 in zip(items0, items1)))

    def test_reiter_once(self):
        # This differs from test_repeat_once in that we use checkpoints.
        reader: Iterable[Any] = _InfinitePermutationIterator(self.flattened_test_data, 42)
        checkpoint = reader.__getstate__()
        items0 = list(itertools.islice(reader, len(self.flattened_test_data)))
        reader.__setstate__(checkpoint)
        items1 = list(itertools.islice(reader, len(self.flattened_test_data)))
        self.assertMultisetEqual(items0 + items1, self.flattened_test_data * 2)
        self.assertSequenceEqual(items0, items1)


class TestChunkedDataIterator(TestBase):    
    def test(self):
        items = list(_IterableChunkedData(self.chunk_file_paths))
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

        lf_data = list(_IterableChunkedData([lf_file]))
        crlf_dat = list(_IterableChunkedData([crlf_file]))
        self.assertListEqual(lf_data, crlf_dat)

        shutil.rmtree(lf_dir)
        shutil.rmtree(crlf_dir)


class TestBufferedShuffleIterator(TestBase):
    def test_shuffle(self):
        items = list(_IterableBufferedShuffler(self.flattened_test_data.copy(), 971, 42))
        self.assertMultisetEqual(items, self.flattened_test_data)

    def test_shuffle_buffer_size_one(self):
        items = list(_IterableBufferedShuffler(self.flattened_test_data.copy(), 1, 42))
        self.assertMultisetEqual(items, self.flattened_test_data)


class TestIterableChunkedDataset(TestBase):
    def test_no_shuffle(self):
        items = list(itertools.islice(IterableChunkedDataset(self.data_dir, shuffle=False), len(self.flattened_test_data)))
        self.assertListEqual(items, self.flattened_test_data)
    
    def test_other_files_present(self):
        with open(os.path.join(self.data_dir, 'i_do_not_belong_here.txt'), 'w') as f:
            f.write('really ...')
        items = list(itertools.islice(IterableChunkedDataset(self.data_dir, shuffle=False), len(self.flattened_test_data)))
        self.assertListEqual(items, self.flattened_test_data)

    def test_transform(self):
        transform = lambda s: s + '!'
        modified_test_data = [transform(s) for s in self.flattened_test_data]
        items = list(itertools.islice(IterableChunkedDataset(self.data_dir, shuffle=False, transform=transform), len(self.flattened_test_data)))
        self.assertListEqual(items, modified_test_data)

    def test_two_instances(self):
        dataset0 = IterableChunkedDataset(self.data_dir, shuffle=False, num_instances=2, instance_rank=0)
        dataset1 = IterableChunkedDataset(self.data_dir, shuffle=False, num_instances=2, instance_rank=1)
        items0 = list(itertools.islice(dataset0, len(self.test_data[0]) + len(self.test_data[2])))
        items1 = list(itertools.islice(dataset1, len(self.test_data[1]) + len(self.test_data[3])))
        self.assertMultisetEqual(set(items0 + items1), self.flattened_test_data)


if __name__ == '__main__':
    unittest.main()