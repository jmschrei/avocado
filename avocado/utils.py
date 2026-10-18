# utils.py
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

"""
This file defines useful utility functions, mostly for the processing of
epigenomic data sets.

On Python 3 this module cannot be imported. The module-level `chroms` list is
built with `range(1, 23) + ['X']`, which raises a TypeError there because
`range` no longer returns a list. `import avocado` does not import this
module, so the rest of the package is unaffected.
"""

import numpy
import pandas
import os

from tqdm import tqdm

chroms = range(1, 23) + ['X']

def download_bigWig(url, download_filepath='.', chroms=chroms, 
	chrom_lengths=None, verbose=True):
	"""Download a bigWig and convert it to one 25 bp resolution array per chromosome.

	This function shells out to `wget` to download `url` into
	`download_filepath`, then, for each chromosome, runs the UCSC tool
	`bigWigToBedGraph` to extract that chromosome into a bedgraph in the
	current working directory, reads it with `bedgraph_to_dense`, averages
	it into 25 bp bins with `decimate_vector`, and deletes the bedgraph.
	`wget` and `bigWigToBedGraph` must be on the PATH, and the commands are
	run with `os.system`, so a failure in either is not raised.

	This function does not return. After the last chromosome it refers to the
	undefined name `bigWig` (the variable is `bigwig`), so it raises a
	NameError, and the downloaded bigWig is not deleted. Passing
	`chrom_lengths` raises a NameError earlier, because the padding step
	refers to the undefined name `chromosome_lengths`.

	Parameters
	----------
	url : str
		The URL of the bigWig file to download.

	download_filepath : str, optional
		The directory to download the bigWig into. Default is '.'.

	chroms : list, optional
		The chromosomes to extract, without the 'chr' prefix. Default is
		chromosomes 1 through 22 and X.

	chrom_lengths : list or None, optional
		The number of 25 bp bins to pad each chromosome's array to, in the
		order of chromosomes 1 through 22 and then X. Default is None.

	verbose : bool, optional
		Whether to print each command before it is run and to show a progress
		bar while reading each bedgraph. Default is True.

	Returns
	-------
	chrom_data : list of numpy.ndarray
		One array per chromosome in `chroms`, holding the mean signal in each
		25 bp bin. This value is never reached; see above.
	"""

	if verbose == True:
		print("wget -nv -q {} -P {}/".format(url, download_filepath))

	os.system("wget -nv {} -P {}/".format(url, download_filepath))

	name = url.split('/')[-1]
	bigwig = '{}/{}'.format(download_filepath, name)
	name = name if '.' not in name else '.'.join(name.split('.')[:-1])

	chrom_data = []

	for chrom in chroms:
		bedgraph = name + '.chr{}.bedgraph'.format(chrom)
			
		if verbose == True:
			print("bigWigToBedGraph {} {} -chrom=chr{}".format(bigwig, 
				bedgraph, chrom))
		
		os.system("bigWigToBedGraph {} {} -chrom=chr{}".format(bigwig, 
			bedgraph, chrom))

		if verbose == True:
			print("bedgraph_to_dense({})".format(bedgraph))
		
		data = bedgraph_to_dense(bedgraph, verbose=verbose)

		if verbose == True:
			print("decimate_vector")

		data = decimate_vector(data)

		if chrom_lengths is not None:
			if chrom != 'X':
				data_ = numpy.zeros(chromosome_lengths[chrom-1])
			else:
				data_ = numpy.zeros(chromosome_lengths[-1])

			data_[:len(data)] = data
			data = data_

		chrom_data.append(data)

		if verbose == True:
			print("rm {}".format(bedgraph))
		
		os.system("rm {}".format(bedgraph))

	if verbose == True:
		print("rm {}".format(bigWig))

	os.system("rm {}".format(bigWig))
	return chrom_data

def bedgraph_to_dense(filename, verbose=True):
	"""Read a bedgraph file and return a dense numpy array.

	This will read in a bedgraph file that has four columns corresponding
	to chrom, start, end, value, and return a numpy array of size equal
	to the last "end" entry in the bedgraph and the appropriate entries filled
	in. This assumes that bedgraphs follow the standard indexing format as 
	described by the UCSC genome browser and so are zero-indexed and half open. 
	This means that the first position is 0 and that the end position is not
	included in the range. For example the line

		chr1	0	4	6.3
		chr1	4	7	7.2
		chr1	7	9	0.2

	Would yield an array like the following:

	array([6.3	6.3	6.3	6.3	7.2 7.2 7.2 0.2 0.2])

	Note that this will return a dense array and so even if the value at many
	positions is equal to 0 they will be explicitly set to 0.

	Parameters
	----------
	filename : str
		The name of the bedgraph file to use.

	verbose : bool, optional
		Whether to return the progress through the file as a progress bar.
		This returns the progress through the bedgraph file not the progress
		through the array, meaning that at the 50% mark, half of the file has
		been read, not half of the array has been filled in. Default is True.

	Returns
	-------
	array : numpy.ndarray, shape=(n,)
		A dense array of the unpacked values.
	"""

	bedgraph = pandas.read_csv(filename, sep="\t", header=None)
	n = bedgraph[2].values[-1]
	k = bedgraph.shape[0]
	data = numpy.zeros(n)

	d = not verbose
	for i, (_, start, end, v) in tqdm(bedgraph.iterrows(), total=k, disable=d):
		data[start:end] = v

	return data

def decimate_vector(x, k=25, func=numpy.mean):
	"""Decimate a vector to reduce its size, returning the average values.

	This function will take in a vector and reduce its size by taking the
	average value of each k-sized window along the genome. This function is 
	used to take the nucleotide resolution epigenomic data provided and convert
	it to 25 bp resolution. k can be modified to any number to yield a finer
	or coarser resolution.

	Parameters
	----------
	x : list or numpy.ndarray
		The vector of values to decimate.

	k : int, optional
		The resolution at which to downsample and decimate. Default is 25,
		indicating that the resulting vector should be 1/25th the size and
		each position in the new vector is derived from 25 values from the
		original vector. 

	func : function, optional
		The function used to aggregate a k-sized window into a single value.

	Returns
	-------
	y : numpy.ndarray, shape=(x.shape[0] // k)
		The decimated array whose size is now 1 // k of the original vector
		and whose values are derived by applying `func` to each of the k-sized
		windows from the original data set.
	"""

	m = x.shape[0] // k
	y = numpy.zeros(m)

	for i in range(m):
		y[i] = func(x[i*k:(i+1)*k])
	
	return y

