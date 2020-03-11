# InfiniBatch

To run unit tests, run the following command.
```
python -m unittest discover -s test
```

When working on the documentation, install pdoc:
```
pip install pdoc3
```
You can then start a local http server that dynamically updates the documentation:
```
pdoc --template-dir docs --http : infinibatch
```