#!/usr/bin/python3.6

# simple command-line wrapper around the ChunkedDataset
# Example:
#   block_randomize my_chunked_data

import os, sys, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))) 

from infinibatch.common.chunked_dataset import ChunkedDataset

sets = sys.argv[1:]

sets = sets[0]  # @TODO: ChunkedDataset should support multiple sets

ds = ChunkedDataset(sets, shuffle=True, buffer_size=10000000) # @TODO: seed=1
for line in ds:
    print(line)
