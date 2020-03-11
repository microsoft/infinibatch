import doctest
import infinibatch.iterators

def load_tests(loader, tests, ignore):
    # include doctests into unittests
    tests.addTests(doctest.DocTestSuite(infinibatch.iterators))
    return tests