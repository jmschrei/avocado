from setuptools import setup

setup(
    name='avocado-epigenome',
    version='0.2.0',
    author='Jacob Schreiber',
    author_email='jmschr@cs.washington.edu',
    packages=['avocado'],
    license='LICENSE.txt',
    description='Avocado is a package for learning a latent representation of the human epigenome.',
    install_requires=[
        "numpy == 1.14.2",
        "keras == 2.0.8",
        "theano == 1.0.1"
    ]
)
