#!/usr/bin/python3.6

# simple command-line wrapper around the chunked_dataset_iterator
# Example:
#   block_randomize my_chunked_data

import os, sys, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))  # find our imports

from infinibatch.datasets import chunked_dataset_iterator

sets = sys.argv[1:]

ds = chunked_dataset_iterator(sets, shuffle=True, buffer_size=1000000, seed=1)
for line in ds:
    print(line)
