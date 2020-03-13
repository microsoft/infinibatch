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

To get started, please look at documentation of the module `iterators`.
"""