from setuptools import setup

with open("README.md", "r") as infile:
	long_description = infile.read()

setup(
	name='avocado-epigenome',
	version='0.4.1',
	author='Jacob Schreiber',
	author_email='jmschr@cs.washington.edu',
	packages=['avocado'],
	scripts=['cli/avocado-impute'],
	license='LICENSE.txt',
	description='Avocado is a package for learning a latent representation of the human epigenome.',
	long_description=long_description,
	long_description_content_type='text/markdown',
	python_requires='<3.9',
	install_requires=[
		"numpy >= 1.14.2, < 1.22",
		"pandas >= 0.23.4",
		"theano >= 1.0.1",
		"keras >= 2.0.8, < 2.4",
		"h5py < 3",
		"tqdm >= 4.24.0",
		"pyBigWig >= 0.3.17"
	]
)
