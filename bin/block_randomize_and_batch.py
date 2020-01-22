#!/usr/bin/python3.6

# simple command-line wrapper around BucketedReadaheadBatchGenerator on a ChunkedDataset
# Example:
#   block_randomize_and_batch my_chunked_data

import os, sys, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))  # find our imports

from infinibatch.common.chunked_dataset import ChunkedDataset
from infinibatch.common.bucketed_readahead_batch_generator import BucketedReadaheadBatchGenerator

sets = sys.argv[1:]

ds = ChunkedDataset(sets, shuffle=True, buffer_size=10000000, seed=1)
batch_labels = 500
bg = BucketedReadaheadBatchGenerator(ds, read_ahead=100, key=lambda line: len(line), batch_size_fn=lambda line: batch_labels // len(line), seed=1)
for batch in bg:
    print(f"\n---- size {len(batch)} ---\n")
    print("\n".join(batch))
