# InfiniBatch

To view the documentation, please clone the repository and go to docs/infinibatch/index.html

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

We currently haven't set up the CI to automatically generate the documentation.
Before you merge anything into master, please delete the existing documentation in docs/infinibatch and run
```
pdoc -o docs --template-dir docs --html infinibatch
```

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
