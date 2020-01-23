import gzip
import itertools
import os
import random
import shutil
import tempfile
import unittest

from infinibatch.common.chunked_dataset import ChunkedDataset, ChunkPermutationIterator, ChunkedDataReader, BufferedShuffleIterator


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


class TestChunkPermutationIterator(TestBase):
    def test_no_shuffle(self):
        items = list(ChunkPermutationIterator(self.chunk_file_paths, False, None))
        self.assertListEqual(items, self.chunk_file_paths)


    def test_shuffle(self):
        items = list(ChunkPermutationIterator(self.chunk_file_paths, False, random.Random(42)))
        self.assertMultisetEqual(items, self.chunk_file_paths)


    def test_repeat_once_no_shuffle(self):
        items = list(itertools.islice(ChunkPermutationIterator(self.chunk_file_paths, True, None), 2 * len(self.chunk_file_paths)))
        self.assertListEqual(items, self.chunk_file_paths * 2)


    def test_repeat_once_shuffle(self):
        reader = ChunkPermutationIterator(self.chunk_file_paths, True, random.Random(42))
        items0 = list(itertools.islice(reader, len(self.chunk_file_paths)))
        items1 = list(itertools.islice(reader, len(self.chunk_file_paths)))

        self.assertMultisetEqual(items0 + items1, self.chunk_file_paths * 2)
        
        different = False
        for item0, item1 in zip(items0, items1):
            if item0 != item1:
                different = True
                break
        self.assertTrue(different)


class TestChunkedDataReader(TestBase):    
    def test(self):
        items = list(ChunkedDataReader(self.chunk_file_paths))
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

        lf_data = list(ChunkedDataReader([lf_file]))
        crlf_dat = list(ChunkedDataReader([crlf_file]))

        self.assertListEqual(lf_data, crlf_dat)

        shutil.rmtree(lf_dir)
        shutil.rmtree(crlf_dir)


class TestBufferedShuffleIterator(TestBase):
    def test_shuffle(self):
        items = list(BufferedShuffleIterator(self.flattened_test_data.copy(), 971))
        self.assertMultisetEqual(items, self.flattened_test_data)


    def test_shuffle_buffer_size_one(self):
        items = list(BufferedShuffleIterator(self.flattened_test_data.copy(), 1))
        self.assertMultisetEqual(items, self.flattened_test_data)


class TestChunkedDataset(TestBase):
    def test_no_shuffle(self):
        items = list(ChunkedDataset(self.data_dir, shuffle=False))
        self.assertListEqual(items, self.flattened_test_data)


    def test_shuffle(self):
        items = list(ChunkedDataset(self.data_dir, shuffle=True))
        self.assertMultisetEqual(items, self.flattened_test_data)

    
    def test_other_files_present(self):
        with open(os.path.join(self.data_dir, 'i_do_not_belong_here.txt'), 'w') as f:
            f.write('really ...')
        items = list(ChunkedDataset(self.data_dir, shuffle=False))
        self.assertListEqual(items, self.flattened_test_data)


    def test_transform(self):
        transform = lambda s: s + '!'
        modified_test_data = [transform(s) for s in self.flattened_test_data]
        items = list(ChunkedDataset(self.data_dir, shuffle=False, transform=transform))
        self.assertListEqual(items, modified_test_data)


if __name__ == '__main__':
    unittest.main()