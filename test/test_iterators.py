import copy
import itertools
import random
import unittest

from infinibatch.iterators import *

if __name__ == "__main__":
    unittest.main()


class TestBase(unittest.TestCase):
    def assertMultisetEqual(self, a, b):
        def list_to_dict(l):
            d = {}
            for item in l:
                d[item] = d.get(item, 0) + 1
            return d

        self.assertEqual(list_to_dict(a), list_to_dict(b))


class TestInfinitePermutationSourceIterator(TestBase):
    def setUp(self):
        self.lengths = [1, 2, 3, 4, 5, 42, 157, 256]
        self.repeats = [1, 2, 3, 4, 5]

    def test_no_shuffle(self):
        for n, k in itertools.product(self.lengths, self.repeats):
            with self.subTest(f"n={n}, k={k}"):
                data = list(range(n))
                it = InfinitePermutationSourceIterator(copy.deepcopy(data), shuffle=False)
                result = [next(it) for _ in range(k * n)]
                self.assertEqual(data * k, result)

    def test_shuffle(self):
        for n, k in itertools.product(self.lengths, self.repeats):
            with self.subTest(f"n={n}, k={k}"):
                data = list(range(n))
                it = InfinitePermutationSourceIterator(copy.deepcopy(data))
                result = [next(it) for _ in range(k * n)]
                self.assertMultisetEqual(data * k, result)

    def test_checkpointing_from_start(self):
        for n, k in itertools.product(self.lengths, self.repeats):
            with self.subTest(f"n={n}, k={k}"):
                data = list(range(n))
                it = InfinitePermutationSourceIterator(copy.deepcopy(data))
                expected_result = [next(it) for _ in range(k * n)]  # extract data
                it.setstate(None)  # reset to start
                result = [next(it) for _ in range(k * n)]
                self.assertEqual(result, expected_result)

    def test_checkpointing_from_middle(self):
        for n, k in itertools.product(self.lengths, self.repeats):
            with self.subTest(f"n={n}, k={k}"):
                data = list(range(n))
                it = InfinitePermutationSourceIterator(copy.deepcopy(data))
                checkpoint_pos = k * n // 3
                for _ in range(checkpoint_pos):  # go to checkpoint_pos
                    next(it)
                checkpoint = it.getstate()  # take checkpoint
                expected_result = [next(it) for _ in range(k * n)]  # extract data
                for _ in range(checkpoint_pos):  # move forward some more
                    next(it)
                it.setstate(checkpoint)  # reset to checkpoint
                result = [next(it) for _ in range(k * n)]  # get data again
                self.assertEqual(result, expected_result)

    def test_checkpointing_at_boundary(self):
        for n, k in itertools.product(self.lengths, self.repeats):
            with self.subTest(f"n={n}, k={k}"):
                data = list(range(n))
                it = InfinitePermutationSourceIterator(copy.deepcopy(data))
                checkpoint_pos = k * n
                for _ in range(checkpoint_pos):  # go to checkpoint_pos
                    next(it)
                checkpoint = it.getstate()  # take checkpoint
                expected_result = [next(it) for _ in range(k * n)]  # extract data
                for _ in range(checkpoint_pos):  # move forward some more
                    next(it)
                it.setstate(checkpoint)  # reset to checkpoint
                result = [next(it) for _ in range(k * n)]  # get data again
                self.assertEqual(result, expected_result)

    # this test currently hangs / fails because of a bug
    # def test_multiple_instances(self):
    #     world_sizes = [1, 2, 3, 4, 5, 11, 16, 128, 255, 774]
    #     for n, k, num_instances in itertools.product(self.lengths, self.repeats, world_sizes):
    #         data = list(range(n))
    #         it = InfinitePermutationSourceIterator(copy.deepcopy(data))
    #         single_instance_data = [next(it) for _ in range(k * n * num_instances)]
    #         for instance_rank in range(num_instances):
    #             with self.subTest(f"n={n}, k={k}, num_instances={num_instances}, instance_rank={instance_rank}"):
    #                 it = InfinitePermutationSourceIterator(
    #                     copy.deepcopy(data), num_instances=num_instances, instance_rank=instance_rank
    #                 )
    #                 expected_data = []
    #                 pos = instance_rank
    #                 while len(expected_data) < k * n:
    #                     expected_data.append(data[pos])
    #                     pos += instance_rank
    #                 result = [next(it) for _ in range(k * n)]
    #                 self.assertEqual(expected_data, result)

    def test_empty_source(self):
        def create_iterator():
            it = InfinitePermutationSourceIterator([])

        self.assertRaises(ValueError, create_iterator)

    def test_rank_too_large(self):
        def create_iterator():
            it = InfinitePermutationSourceIterator([1], num_instances=2, instance_rank=2)

        self.assertRaises(ValueError, create_iterator)


class TestChunkedSourceIterator(TestBase):
    # ChunkedSourceIterator has no custom checkpointing logic
    # so we do not test checkpointing here
    def test(self):
        lengths = [1, 2, 3, 4, 5, 42, 157, 256]
        world_sizes = [1, 2, 3, 4, 5, 11, 16, 128, 255, 774]
        for n, num_instances in itertools.product(lengths, world_sizes):
            with self.subTest(f"n={n}, num_instances={num_instances}"):
                data = list(range(n))
                result = []
                sizes = []
                for instance_rank in range(num_instances):
                    it = ChunkedSourceIterator(
                        copy.deepcopy(data), num_instances=num_instances, instance_rank=instance_rank
                    )
                    output = list(it)
                    result.extend(output)
                    sizes.append(len(output))
                self.assertEqual(data, result)
                self.assertTrue(max(sizes) - min(sizes) <= 1)  # make sure data is split as evenly as possible

    def test_rank_too_large(self):
        def create_iterator():
            it = ChunkedSourceIterator([1], num_instances=2, instance_rank=2)

        self.assertRaises(ValueError, create_iterator)


class TestSamplingRandomMapIterator(TestBase):
    def setUp(self):
        self.lengths = [1, 2, 3, 4, 5, 42, 157, 256]
        self.seed = 42

    @staticmethod
    def transform(random, item):
        return item + random.random()

    def test_basic(self):
        for n in self.lengths:
            data = list(range(n))
            random = Random()
            random.seed(self.seed)
            expected_result = [n + random.random() for n in data]
            it = SamplingRandomMapIterator(
                NativeCheckpointableIterator(copy.deepcopy(data)), transform=self.transform, seed=self.seed
            )
            result = list(it)
            self.assertEqual(result, expected_result)

    def test_checkpointing_from_start(self):
        for n in self.lengths:
            data = list(range(n))
            it = SamplingRandomMapIterator(
                NativeCheckpointableIterator(copy.deepcopy(data)), transform=self.transform, seed=self.seed
            )
            expected_result = list(it)  # extract data
            it.setstate(None)  # reset to start
            result = list(it)
            self.assertEqual(result, expected_result)

    def test_checkpointing_from_middle(self):
        for n in self.lengths:
            data = list(range(n))
            it = SamplingRandomMapIterator(
                NativeCheckpointableIterator(copy.deepcopy(data)), transform=self.transform, seed=self.seed
            )
            checkpoint_pos = n // 3
            for _ in range(checkpoint_pos):  # go to checkpoint_pos
                next(it)
            checkpoint = it.getstate()  # take checkpoint
            expected_result = list(it)  # extract data
            it.setstate(checkpoint)  # reset to checkpoint
            result = list(it)
            self.assertEqual(result, expected_result)
