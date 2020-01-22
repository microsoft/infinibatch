#!/usr/bin/python3.6

# simple command-line wrapper around the ChunkedDataset
# Example:
#   block_randomize my_chunked_data

import os, sys, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))  # find our imports

from infinibatch.common.chunked_dataset import ChunkedDataset

sets = sys.argv[1:]

ds = ChunkedDataset(sets, shuffle=True, buffer_size=10000000, seed=1)
for line in ds:
    print(line)
