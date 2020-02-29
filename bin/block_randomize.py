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
    paths = sys.argv[3:]
else:
    credential = None
    paths = sys.argv[1:]

def readlines_from_zipped(textfile_path: str):
    #print("chunked_dataset_iterator: reading", textfile_path, file=sys.stderr)
    return iter(read_utf8_file(textfile_path, credential).splitlines())

if isinstance(paths, str):  # handle single string
    paths = [paths]
chunk_file_paths = [  # enumerate all .gz files in the given paths
    subpath
    for path in paths
    for subpath in find_files(path, '.gz', credential)
]
chunk_file_paths.sort()  # make sure file order is always the same, independent of OS
#print("chunked_dataset_iterator: reading from", len(chunk_file_paths), "chunk files", file=sys.stderr)

ds = chunked_dataset_iterator(chunk_file_paths, read_chunk_fn=readlines_from_zipped, shuffle=True, buffer_size=1000000, seed=1)
for line in ds:
    print(line)
