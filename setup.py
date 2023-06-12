from setuptools import setup, find_packages

long_description = open('README.md', encoding='utf-8').read()
classifiers = [  # copied from https://pypi.org/classifiers/
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Topic :: Utilities',
    'Topic :: Text Processing',
    'Topic :: Text Processing :: General',
    'Topic :: Text Processing :: Filters',
    'Topic :: Text Processing :: Linguistic',
    'Programming Language :: Python :: 3 :: Only',
]

setup(
    name='infinibatch',
    version='0.1.1',
    url='https://github.com/microsoft/infinibatch',
    author='Frank Seide',
    author_email='fseide@microsoft.com',
    description='Infinibatch is a library of checkpointable iterators for randomized data loading of massive data sets in deep neural network training.',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    platforms=['any'],
    python_requires='>=3.6',
    keywords=['datasets', 'NLP', 'natural language processing,' 'computational linguistics'],
    download_url='https://github.com/microsoft/infinibatch',
)
