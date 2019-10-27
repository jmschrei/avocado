from setuptools import setup

setup(
    name='avocado-epigenome',
    version='0.3.2',
    author='Jacob Schreiber',
    author_email='jmschr@cs.washington.edu',
    packages=['avocado'],
    scripts=['cli/avocado-impute'],
    license='LICENSE.txt',
    description='Avocado is a package for learning a latent representation of the human epigenome.',
    install_requires=[
        "numpy >= 1.14.2",
        "pandas >= 0.23.4",
        "theano >= 1.0.1",
        "keras >= 2.0.8",
        "tqdm >= 4.24.0"
    ]
)
