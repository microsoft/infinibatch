"""
Infinibatch is a library of checkpointable iterators for randomized data loading of massive data sets in deep neural network training.


# Features

  * support for corpora much larger than fit into RAM
  * hierarchical block+sentence-level randomization over the whole corpus, different randomization in each epoch
  * only load the data that is needed
  * very fast start-up time (does not need to read full corpus)
  * only requires the most basic of data preparation (e.g. no indexing)
  * for multi-GPU, only load what the respective GPU needs
  * 100% accurate check-pointing, restore from checkpoint should not read all data up to the checkpoint
  * support automatic bucketed batching with dynamic batch sizes
  * pre-fetching thread
  * composable, as to support for complex batching, e.g. negative samples from multiple documents


# Getting Started

To install `infinibatch`, please copy this library into a subfolder in your project:
```
cd YOUR_PROJECT_FOLDER
git clone https://msasg.visualstudio.com/DefaultCollection/SDRG/_git/infinibatch
```
or, better, as a submodule reference:
```
git submodule add https://msasg.visualstudio.com/DefaultCollection/SDRG/_git/infinibatch
```
It is now located at `infinibatch/infinibatch`, e.g. the main import file is `infinibatch/infinibatch/__init__.py`.

To import it, you need to add that folder to your `PYTHONPATH` variable externally, or to `sys.path` inside the code:
```
import sys
sys.path.insert(0,'infinibatch')  # note: relative paths are relative to your current dir, not to the python script
import infinibatch
```
There are no further dependencies. Infinibatch requires Python 3.6 or higher.

# Tutorial

This little tutorial walks through the steps of preparing your data and consuming them from Python code as batches.

## Infinibatch Basics: Iterators and Checkpointing

Infinibatch provides [Python iterators](https://docs.python.org/3.5/glossary.html#term-iterator)
to read your data.
An iterator represents a stream of data that can be retrieved item by item, e.g. via a
`for` loop or repeated calls to the iterator's `__next__()` method.
An item can be, for example, a pair of lines of text, or an audio file
with a textual annotation.

Infinibatch makes it easy to read your data in randomized order, with checkpointing.

Randomization is _on the fly_, which means that you don't need to read the entire data set into memory
to be shuffled. Infinibatch implements a hierarchical shuffling algorithm
that only holds a subset of the data in RAM at any point in time.

Infinibatch iterators are _checkpointable_. The sad reaslity is that long-running trainings occasionally crash.
Checkpointing lets you retrieve the current position (the "checkpoint") in the data stream at any time, so that
later, you can "rewind" to that same position.
If you save your Infinibatch iterator's checkpoint to disk whenever you save an intermediate model during training,
then if your training crashes, you can load the checkpoint and continue your training
on exactly the same data-item sequence you would have trained on without the crash.

## Data Preparation

Infinibatch has one requirement on your data organization:
To use your data with Infnibatch, it must be split into a large number of small chunks.
A chunk is the smallest unit that is loaded from disk. Infinibatch holds a random subset of chunks in memory
that it randomly draws samples from.

We will show how such split can be created An easy way to split your data into chunks is with the Linux `split` command.

Let us create a simple test file first, which will act as the corpus in this tutorial.
In a bash shell, please run this command:
```
echo \\
'Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
The quick brown fox jumps over the lazy dog.' \\
> corpus.txt
```
Now let us split it into 3 chunks of 2 lines each, which is zipped. We will create them inside a
new subdirectory called `corpus_chunks`:
```
mkdir corpus_chunks
split  --lines 2  --numeric-suffixes  --filter 'gzip > corpus_chunks/$FILE.txt.gz'  corpus.txt  corpus.
```
This will have created three files: `corpus_chunks/corpus.00.txt.gz`, `corpus_chunks/corpus.01.txt.gz`, and `corpus_chunks/corpus.02.txt.gz`.
To verify whether the data has been split as expected, you can use this command: `zcat corpus_chunks/corpus.*.txt.gz`.

Hint: For large corpora, we recommend replacing `gzip` by `pigz`, which runs notably faster via multi-threading.

## Reading items in random order with Infinibatch

We will first show the easiest way to read data with Infinibatch, using a helper function `chunked_dataset_iterator()`.
This function will create an Infinibatch iterator that yields the content of your data in random order.
If you run the following program:
```
import sys, gzip, glob
sys.path.insert(0,'infinibatch')
from infinibatch import datasets as ds

ds = ds.chunked_dataset_iterator(
    chunk_refs = glob.glob('corpus_chunks/corpus.*.txt.gz'),
    read_chunk_fn = lambda path: iter(gzip.decompress(open(path, "rb").read()).decode(encoding='utf-8').splitlines()),
    buffer_size = 6, seed = 1)

for i in range(10):
    print(next(ds))
```
you should get output that contains the 6 example lines in randomized order:
```
Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
The quick brown fox jumps over the lazy dog.
sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
consectetur adipiscing elit,
Lorem ipsum dolor sit amet,
The quick brown fox jumps over the lazy dog.
sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
```
Note: The `buffer_size` parameter determines how many sentences are read into memory at any given time,
for being randomized. In real settings with corpora of hundreds of millions of text lines,
the `buffer_size` parameter should be set in the millions.

## Reading items of different lengths in batches

For deep learning, we want to group multiple items into batches.
For text tasks, items are often of different lengths.
Infinibatch implements an algorithm that randomizes the input sequence and groups it into
batches of approximately the same length.

Infinibatch's `BucketedReadaheadBatchIterator()` performs this task.
Here is an example. Please note that `BucketedReadaheadBatchIterator()` takes the previous
randomized sentence sequence iterator in `ds` as an argument.
```
import sys, gzip, glob
sys.path.insert(0,'infinibatch')
from infinibatch import datasets as ds
from infinibatch import iterators as it

ds = ds.chunked_dataset_iterator(
    chunk_refs = glob.glob('corpus_chunks/corpus.*.txt.gz'),
    read_chunk_fn = lambda path: iter(gzip.decompress(open(path, "rb").read()).decode(encoding='utf-8').splitlines()),
    buffer_size = 6, seed = 1)

bs = it.BucketedReadaheadBatchIterator(
    source_iterator = ds,
    read_ahead = 6,
    key = lambda line: len(line),
    batch_size = 2,
    seed = 1)

for i in range(25):
    print(next(bs))
```
This code should output something like this:
```
['sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.', 'The quick brown fox jumps over the lazy dog.']
['consectetur adipiscing elit,', 'Lorem ipsum dolor sit amet,']
['Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.', 'Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.']
```
followed by different permutations of the same tuples.
As you can see, the sentences are in random order and grouped in batches of 2 of approximately the same length.
In this examples, all permutations have the same batch groupings. In real examples, where the data size is much larger
than the batch size, this will not be the case.

Infinibatch can also help batching variable-size batches. Please change the `batch_size` parameter to a lambda
that determines the number of items as a function of the length of the longest item in a batch:
```
    batch_size = lambda line: 150 // len(line),
```
The output looks like this:
```
['consectetur adipiscing elit,', 'Lorem ipsum dolor sit amet,']
['Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.']
['sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.', 'The quick brown fox jumps over the lazy dog.']
['Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.']
```
That shorter sentences got grouped, while longer did not because they would exceed the total of 150.

## Reading batches into numpy arrays

Lastly, we will need to feed batches into our favorite deep-learning tool.
The following code converts the input lines of each batch into a sequence
of character ASCII codes (a real scenario would have word indices here).
These sequences are then padded to equal length with -1, and converted into a `numpy` array.

Please rerun the previous example, but first insert the following code before the final `for` loop:
```
import numpy as np
def collate(lines_batch):
    ids_batch = [[ord(c) for c in line] for line in lines_batch]
    width = max(len(ids) for ids in ids_batch)
    return np.array([ids + [-1] * (width-len(ids)) for ids in ids_batch])

bs = it.MapIterator(
    source_iterator = bs,
    transform = collate)
```
This will output batches like this. Note that in batches with multiple sentences,
some entries are padded with `-1`.
```
[[ 99 111 110 115 101  99 116 101 116 117 114  32  97 100 105 112 105 115
   99 105 110 103  32 101 108 105 116  44]
 [ 76 111 114 101 109  32 105 112 115 117 109  32 100 111 108 111 114  32
  115 105 116  32  97 109 101 116  44  -1]]
[[ 85 116  32 101 110 105 109  32  97 100  32 109 105 110 105 109  32 118
  101 110 105  97 109  44  32 113 117 105 115  32 110 111 115 116 114 117
  100  32 101 120 101 114  99 105 116  97 116 105 111 110  32 117 108 108
   97 109  99 111  32 108  97  98 111 114 105 115  32 110 105 115 105  32
  117 116  32  97 108 105 113 117 105 112  32 101 120  32 101  97  32  99
  111 109 109 111 100 111  32  99 111 110 115 101 113 117  97 116  46]]
[[115 101 100  32 100 111  32 101 105 117 115 109 111 100  32 116 101 109
  112 111 114  32 105 110  99 105 100 105 100 117 110 116  32 117 116  32
  108  97  98 111 114 101  32 101 116  32 100 111 108 111 114 101  32 109
   97 103 110  97  32  97 108 105 113 117  97  46]
 [ 84 104 101  32 113 117 105  99 107  32  98 114 111 119 110  32 102 111
  120  32 106 117 109 112 115  32 111 118 101 114  32 116 104 101  32 108
   97 122 121  32 100 111 103  46  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1
   -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1]]
[[ 68 117 105 115  32  97 117 116 101  32 105 114 117 114 101  32 100 111
  108 111 114  32 105 110  32 114 101 112 114 101 104 101 110 100 101 114
  105 116  32 105 110  32 118 111 108 117 112 116  97 116 101  32 118 101
  108 105 116  32 101 115 115 101  32  99 105 108 108 117 109  32 100 111
  108 111 114 101  32 101 117  32 102 117 103 105  97 116  32 110 117 108
  108  97  32 112  97 114 105  97 116 117 114  46]]
```

# Where To Go From Here

The above tutorial showed you the use of the most common iterator type, as created by the
convenience function `chunked_dataset_iterator()`.

Not all real-life scenarios are covered by this function. For example, multi-task learning
scenarios require more complex combinations of data. To create those, you will need
to compose the necessary data reader from the underlying building blocks.
This is described at the documentation of the module `iterators`.
"""
