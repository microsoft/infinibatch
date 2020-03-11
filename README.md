# InfiniBatch

To run unit tests, run the following command.
```
python -m unittest discover -s test
```

To run doctests for the iterators, run the following command.
```
python -m doctest -v infinibatch\iterators.py
```

When working on the documentation, install pdoc:
```
pip install pdoc3
```
You can then start a local http server that dynamically updates the documentation:
```
pdoc --template-dir docs --http : infinibatch
```