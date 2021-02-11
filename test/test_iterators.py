import copy
import itertools
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
        self.lengths = [1, 2, 3, 4, 42, 57]
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
                expected_data = [next(it) for _ in range(k * n)]  # extract data
                it.setstate(None)  # reset to start
                result = [next(it) for _ in range(k * n)]
                self.assertEqual(expected_data, result)

    def test_checkpointing_from_middle(self):
        for n, k in itertools.product(self.lengths, self.repeats):
            with self.subTest(f"n={n}, k={k}"):
                data = list(range(n))
                it = InfinitePermutationSourceIterator(copy.deepcopy(data))
                checkpoint_pos = k * n // 3
                for _ in range(checkpoint_pos):  # go to checkpoint_pos
                    next(it)
                checkpoint = it.getstate()  # take checkpoint
                expected_data = [next(it) for _ in range(k * n)]  # extract data
                for _ in range(checkpoint_pos):  # move forward some more
                    next(it)
                it.setstate(checkpoint)  # reset to checkpoint
                result = [next(it) for _ in range(k * n)]  # get data again
                self.assertEqual(expected_data, result)

    def test_checkpointing_at_boundary(self):
        for n, k in itertools.product(self.lengths, self.repeats):
            with self.subTest(f"n={n}, k={k}"):
                data = list(range(n))
                it = InfinitePermutationSourceIterator(copy.deepcopy(data))
                checkpoint_pos = k * n
                for _ in range(checkpoint_pos):  # go to checkpoint_pos
                    next(it)
                checkpoint = it.getstate()  # take checkpoint
                expected_data = [next(it) for _ in range(k * n)]  # extract data
                for _ in range(checkpoint_pos):  # move forward some more
                    next(it)
                it.setstate(checkpoint)  # reset to checkpoint
                result = [next(it) for _ in range(k * n)]  # get data again
                self.assertEqual(expected_data, result)
