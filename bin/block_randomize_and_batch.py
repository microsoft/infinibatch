#!/usr/bin/python3.6

# simple command-line wrapper around the ChunkedDataset
# Example:
#   block_randomize my_chunked_data

import os, sys, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))) 

from infinibatch.common.chunked_dataset import ChunkedDataset
from infinibatch.bucketed_readahead_batch_generator import BucketedReadaheadBatchGenerator

sets = sys.argv[1:]

sets = sets[0]  # @TODO: ChunkedDataset should support multiple sets

ds = ChunkedDataset(sets, shuffle=True, buffer_size=10000000) # @TODO: seed=1
batch_labels = 500
bg = BucketedReadaheadBatchGenerator(ds, read_ahead=100, key=lambda line: len(line), batch_size_fn=lambda line: batch_labels // len(line))
for batch in bg:
    print(f"\n---- size {len(batch)} ---\n")
    print("\n".join(batch))
