import gzip
import os
import shutil
import tempfile
import unittest

import context  # allows us to directly import from our package without installing it first
from infinibatch.common.chunked_dataset import ChunkedDataset, chunked_data_generator, buffered_shuffle_generator, transform_generator


class TestChunkedDataset(unittest.TestCase):
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
            file_content = '\n'.join(chunk).encode('utf-8')
            with gzip.open(file_name, 'w') as f:
                f.write(file_content)


    def tearDown(self):
        shutil.rmtree(self.data_dir)


    def test_chunked_data_generator_no_shuffle(self):
        items = []
        for item in chunked_data_generator(self.chunk_file_paths, shuffle_chunks=False):
            items.append(item)
        self.assertListEqual(items, self.flattened_test_data)


    def test_chunked_data_generator_shuffle(self):
        items = []
        for item in chunked_data_generator(self.chunk_file_paths, shuffle_chunks=True):
            items.append(item)
        self.assertSetEqual(set(items), set(self.flattened_test_data))


    def test_buffered_shuffle_generator(self):
        items = []
        for item in buffered_shuffle_generator(self.flattened_test_data.copy(), 971):
            items.append(item)
        self.assertSetEqual(set(items), set(self.flattened_test_data))


    def test_buffered_shuffle_generator_buffer_size_one(self):
        items = []
        for item in buffered_shuffle_generator(self.flattened_test_data.copy(), 1):
            items.append(item)
        self.assertSetEqual(set(items), set(self.flattened_test_data))


    def test_buffered_shuffle_generator_zero_buffer(self):
        self.assertRaises(ValueError, buffered_shuffle_generator([1], 0).__next__)


    def test_transform_generator(self):
        transform = lambda s: s + '!'
        modified_test_data = [transform(s) for s in self.flattened_test_data]
        items = []
        for item in transform_generator(self.flattened_test_data, transform=transform):
            items.append(item)
        self.assertListEqual(items, modified_test_data)

    
    def test_transform_generator_no_transform(self):
        items = []
        for item in transform_generator(self.flattened_test_data):
            items.append(item)
        self.assertListEqual(items, self.flattened_test_data)


    def test_no_shuffle(self):
        dataset = ChunkedDataset(self.data_dir, shuffle=False)
        items = []
        for item in dataset:
            items.append(item)
        self.assertListEqual(items, self.flattened_test_data)


    def test_shuffle(self):
        dataset = ChunkedDataset(self.data_dir, shuffle=True)
        items = []
        for item in dataset:
            items.append(item)
        self.assertSetEqual(set(items), set(self.flattened_test_data))

    
    def test_other_files_present(self):
        with open(os.path.join(self.data_dir, 'i_do_not_belong_here.txt'), 'w') as f:
            f.write('really ...')

        dataset = ChunkedDataset(self.data_dir, shuffle=False)
        items = []
        for item in dataset:
            items.append(item)
        self.assertListEqual(items, self.flattened_test_data)


    def test_transform(self):
        transform = lambda s: s + '!'
        modified_test_data = [transform(s) for s in self.flattened_test_data]
        dataset = ChunkedDataset(self.data_dir, shuffle=False, transform=transform)
        items = []
        for item in dataset:
            items.append(item)
        self.assertListEqual(items, modified_test_data)


if __name__ == '__main__':
    unittest.main()