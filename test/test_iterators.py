import copy
import itertools
from random import Random
import unittest

from infinibatch.iterators import *

if __name__ == "__main__":
    unittest.main()


class TestBase(unittest.TestCase):
    def setUp(self):
        self.lengths = [1, 2, 3, 4, 5, 42, 157, 256, 997]
        self.world_sizes = [1, 2, 3, 4, 5, 11, 16, 128, 255, 774]
        self.seed = 42

    def assertMultisetEqual(self, a, b):
        def list_to_dict(l):
            d = {}
            for item in l:
                d[item] = d.get(item, 0) + 1
            return d

        self.assertEqual(list_to_dict(a), list_to_dict(b))


class TestFiniteIteratorMixin:
    """
    Mixin to be used in combination with TestBase
    to test basic properties of finite CheckpointableIterators
    """

    def test_basic(self):
        for case_name, expected_result, it in self.test_cases:
            with self.subTest(case_name):
                result = list(it)
                self.assertEqual(result, expected_result)

    def test_checkpointing_reset(self):
        for case_name, expected_result, it in self.test_cases:
            with self.subTest(case_name):
                expected_result = list(it)  # extract data
                it.setstate(None)  # reset to start
                result = list(it)
                self.assertEqual(result, expected_result)

    def test_checkpointing_from_start(self):
        for case_name, expected_result, it in self.test_cases:
            with self.subTest(case_name):
                checkpoint = it.getstate()
                expected_result = list(it)  # extract data
                it.setstate(checkpoint)  # reset to start
                result = list(it)
                self.assertEqual(result, expected_result)

    def _test_checkpointing_from_pos(self, it, pos):
        for _ in range(pos):  # go to pos
            next(it)
        checkpoint = it.getstate()  # take checkpoint
        expected_result = list(it)  # extract data
        it.setstate(checkpoint)  # reset to checkpoint
        result = list(it)
        self.assertEqual(result, expected_result)

    def test_checkpointing_from_one(self):
        for case_name, expected_result, it in self.test_cases:
            with self.subTest(case_name):
                pos = 1
                self._test_checkpointing_from_pos(it, pos)

    def test_checkpointing_from_quarter(self):
        for case_name, expected_result, it in self.test_cases:
            with self.subTest(case_name):
                pos = len(expected_result) // 4
                self._test_checkpointing_from_pos(it, pos)

    def test_checkpointing_from_third(self):
        for case_name, expected_result, it in self.test_cases:
            with self.subTest(case_name):
                pos = len(expected_result) // 3
                self._test_checkpointing_from_pos(it, pos)

    def test_checkpointing_from_half(self):
        for case_name, expected_result, it in self.test_cases:
            with self.subTest(case_name):
                pos = len(expected_result) // 2
                self._test_checkpointing_from_pos(it, pos)

    def test_checkpointing_before_end(self):
        for case_name, expected_result, it in self.test_cases:
            with self.subTest(case_name):
                pos = len(expected_result) - 1
                self._test_checkpointing_from_pos(it, pos)

    def test_checkpointing_at_end(self):
        for case_name, expected_result, it in self.test_cases:
            with self.subTest(case_name):
                list(it)  # exhaust iterator
                checkpoint = it.getstate()  # take checkpoint
                it.setstate(None)  # reset to beginning
                it.setstate(checkpoint)  # reset to checkpoint
                self.assertRaises(StopIteration, it.__next__)

    def test_checkpointing_complex(self):
        for case_name, expected_result, it in self.test_cases:
            with self.subTest(case_name):
                # getstate from fresh iterator
                it.getstate()
                result = list(it)
                self.assertEqual(result, expected_result)

                # get a bunch of checkpoints at different positions
                it.setstate(None)
                positions = [
                    0,
                    len(expected_result) // 7,
                    len(expected_result) // 6,
                    len(expected_result) // 5,
                    len(expected_result) // 4,
                    len(expected_result) // 3,
                    len(expected_result) // 2,
                ]
                checkpoints = []
                for i in range(len(positions)):
                    offset = positions[i] - positions[i - 1] if i > 0 else positions[0]
                    for _ in range(offset):
                        next(it)
                    checkpoints.append(it.getstate())

                # check that iterator returns correct result at all checkpoints
                for pos, checkpoint in zip(positions, checkpoints):
                    it.setstate(checkpoint)
                    self.assertEqual(list(it), expected_result[pos:])

                # check that iterator returns correct result at all checkpoints in reverse order
                tuples = list(zip(positions, checkpoints))
                tuples.reverse()
                for pos, checkpoint in tuples:
                    it.setstate(checkpoint)
                    self.assertEqual(list(it), expected_result[pos:])

                # check that iterator returns correct result at all checkpoints
                # while resetting between any two checkpoints
                for pos, checkpoint in zip(positions, checkpoints):
                    it.setstate(None)
                    it.setstate(checkpoint)
                    self.assertEqual(list(it), expected_result[pos:])

                # and as the grand finale: reset and check again
                it.setstate(None)
                result = list(it)
                self.assertEqual(result, expected_result)


class TestInfinitePermutationSourceIterator(TestBase):
    def setUp(self):
        super().setUp()
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


class TestChunkedSourceIterator(TestBase, TestFiniteIteratorMixin):
    def setUp(self):
        super().setUp()
        self.test_cases = []
        for n in self.lengths:
            data = list(range(n))
            it = ChunkedSourceIterator(copy.deepcopy(data))
            self.test_cases.append((f"n={n}", data, it))

    def test_multiple_instances(self):
        for n, num_instances in itertools.product(self.lengths, self.world_sizes):
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


class TestSamplingRandomMapIterator(TestBase, TestFiniteIteratorMixin):
    @staticmethod
    def transform(random, item):
        return item + random.random()

    def setUp(self):
        super().setUp()
        self.test_cases = []
        for n in self.lengths:
            data = list(range(n))
            random = Random()
            random.seed(self.seed)
            expected_result = [n + random.random() for n in data]
            it = SamplingRandomMapIterator(NativeCheckpointableIterator(data), transform=self.transform, seed=self.seed)
            self.test_cases.append((f"n={n}", expected_result, it))


class TestMapIterator(TestBase, TestFiniteIteratorMixin):
    @staticmethod
    def transform(item):
        return 2 * item

    def setUp(self):
        super().setUp()
        self.test_cases = []
        for n in self.lengths:
            data = list(range(n))
            expected_result = [self.transform(item) for item in data]
            it = MapIterator(NativeCheckpointableIterator(data), self.transform)
            self.test_cases.append((f"n={n}", expected_result, it))


class TestZipIterator(TestBase, TestFiniteIteratorMixin):
    def setUp(self):
        super().setUp()
        self.test_cases = []
        for n in self.lengths:
            data1 = list(range(n))
            data2 = [item * item for item in data1]
            expected_result = list(zip(data1, data2))
            it = ZipIterator(NativeCheckpointableIterator(data1), NativeCheckpointableIterator(data2))
            self.test_cases.append((f"n={n}, pairs", expected_result, it))
        for n in self.lengths:
            data1 = list(range(n))
            data2 = [item * item for item in data1]
            data3 = [item * item for item in data2]
            expected_result = list(zip(data1, data2, data3))
            it = ZipIterator(
                NativeCheckpointableIterator(data1),
                NativeCheckpointableIterator(data2),
                NativeCheckpointableIterator(data3),
            )
            self.test_cases.append((f"n={n}, triples", expected_result, it))

