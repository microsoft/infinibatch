#!/usr/bin/python3.6

# simple command-line wrapper around the chunked_dataset_iterator
# Example:
#   block_randomize my_chunked_data_folder/
#   block_randomize --azure-storage-key $MY_KEY https://myaccount.blob.core.windows.net/mycontainer/my_chunked_data_folder

import os, sys, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))  # find our imports

from infinibatch.datasets import chunked_dataset_iterator
from infinibatch.files_and_blobs import find_files, read_utf8_file

if sys.argv[1] == "--azure-storage-key":
    credential = sys.argv[2]
    sets = sys.argv[3:]
else:
    credential = None
    sets = sys.argv[1:]

def readlines_from_zipped(textfile_path: str) -> Iterable[str]:
    #print("chunked_dataset_iterator: reading", textfile_path, file=sys.stderr)
    return iter(read_utf8_file(textfile_path, credential).splitlines())

ds = chunked_dataset_iterator(sets, read_chunk_fn=readlines_from_zipped, shuffle=True, buffer_size=1000000, seed=1, credentials=credential)
for line in ds:
    print(line)
