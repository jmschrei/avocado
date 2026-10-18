# io.py
# Contact: Jacob Schreiber <jmschr@cs.washington.edu>

"""
This file has the generators that turn a dictionary of epigenomic tracks into
the batches used to train and validate an Avocado model. Each example in a batch is one
(cell type, assay, 25 bp genomic bin) triplet and its signal value, with the
bin also given at 250 bp and 5 kbp resolution for the coarser genome factors.
The generators differ only in how they choose the genomic bins.
"""

import numpy

def data_generator(celltypes, assays, data, n_positions, batch_size):
	"""Yield batches of randomly drawn genomic bins and experiments forever.

	Each example in a batch is drawn independently: a 25 bp genomic bin is drawn
	uniformly at random, with replacement, from the `n_positions` bins, and an
	experiment is drawn uniformly at random from the keys of `data`. There is
	no guarantee on how often each bin is seen. `Avocado.fit`, `fit_celltypes`
	and `fit_assays` use this generator for the validation set.

	This generator indexes `data.keys()` by position, which works on Python 2.
	On Python 3 `dict.keys()` returns a view that cannot be indexed, so the
	first batch raises a `TypeError`, and passing `X_valid` to any of the fit
	methods fails at the end of the first epoch, when Keras first draws a
	validation batch.

	Parameters
	----------
	celltypes : list
		The cell types in the model, in the order of the cell type embedding.
		Each example's cell type is encoded as its index in this list.

	assays : list
		The assays in the model, in the order of the assay embedding. Each
		example's assay is encoded as its index in this list.

	data : dict
		A dictionary where the keys are (celltype, assay) tuples and the values
		are tracks, either numpy arrays in memory or memory maps, each with at
		least `n_positions` entries.

	n_positions : int
		The number of 25 bp genomic bins to draw from.

	batch_size : int
		The number of examples in each batch.

	Yields
	------
	X : dict
		A dictionary mapping the five input layer names ('celltype_input',
		'assay_input', 'genome_25bp_input', 'genome_250bp_input' and
		'genome_5kbp_input') to integer arrays of shape (batch_size,). The
		250 bp and 5 kbp indexes are the 25 bp index divided by 10 and 200.

	y : numpy.ndarray, shape=(batch_size,)
		The signal value of each example's track at its 25 bp bin, as float64.
	"""

	while True:
		celltype_idxs      = numpy.zeros(batch_size, dtype='int32')
		assay_idxs         = numpy.zeros(batch_size, dtype='int32')
		genomic_25bp_idxs  = numpy.random.randint(n_positions, size=batch_size)
		genomic_250bp_idxs = genomic_25bp_idxs // 10
		genomic_5kbp_idxs  = genomic_25bp_idxs // 200
		value              = numpy.zeros(batch_size)

		keys = data.keys()
		idxs = numpy.random.randint(len(data), size=batch_size)

		for i, idx in enumerate(idxs):
			celltype, assay = keys[idx]
			track = data[(celltype, assay)]

			celltype_idxs[i] = celltypes.index(celltype)
			assay_idxs[i]    = assays.index(assay)
			value[i]         = track[genomic_25bp_idxs[i]]

		d = {
			'celltype_input': celltype_idxs, 
			'assay_input': assay_idxs, 
			'genome_25bp_input': genomic_25bp_idxs, 
			'genome_250bp_input': genomic_250bp_idxs,
			'genome_5kbp_input': genomic_5kbp_idxs
		}

		yield d, value

def permuted_data_generator(celltypes, assays, data, n_positions, batch_size):
	"""Yield batches that scan a fixed random permutation of the genomic bins.

	One random permutation of the `n_positions` bins is drawn when the
	generator is created. Successive batches take the next `batch_size` bins of
	that permutation, wrapping around at the end, so every bin is seen once per
	`n_positions / batch_size` batches while the bins in one batch are spread
	across the whole region. The experiment for each bin is drawn uniformly at
	random from the keys of `data`, so a pass over the bins sees one experiment
	per bin rather than all of them.

	`Avocado.fit` uses this generator for both `sampling='permuted'` and
	`sampling='random'`.

	Parameters
	----------
	celltypes : list
		The cell types in the model, in the order of the cell type embedding.
		Each example's cell type is encoded as its index in this list.

	assays : list
		The assays in the model, in the order of the assay embedding. Each
		example's assay is encoded as its index in this list.

	data : dict
		A dictionary where the keys are (celltype, assay) tuples and the values
		are tracks, either numpy arrays in memory or memory maps, each with at
		least `n_positions` entries.

	n_positions : int
		The number of 25 bp genomic bins to draw from.

	batch_size : int
		The number of examples in each batch.

	Yields
	------
	X : dict
		A dictionary mapping the five input layer names ('celltype_input',
		'assay_input', 'genome_25bp_input', 'genome_250bp_input' and
		'genome_5kbp_input') to integer arrays of shape (batch_size,). The
		250 bp and 5 kbp indexes are the 25 bp index divided by 10 and 200.

	y : numpy.ndarray, shape=(batch_size,)
		The signal value of each example's track at its 25 bp bin, as float64.
	"""

	start = 0

	indices = numpy.array([[celltypes.index(celltype) for celltype, _ in data.keys()],
		                   [assays.index(assay) for _, assay in data.keys()]])

	tracks = list(data.values())
	
	permuted_idxs = numpy.arange(n_positions)
	numpy.random.shuffle(permuted_idxs)

	while True:
		celltype_idxs      = numpy.zeros(batch_size, dtype='int32')
		assay_idxs         = numpy.zeros(batch_size, dtype='int32')
		genomic_25bp_idxs  = permuted_idxs[numpy.arange(start, start+batch_size) % n_positions]
		genomic_250bp_idxs = genomic_25bp_idxs // 10
		genomic_5kbp_idxs  = genomic_25bp_idxs // 200
		value              = numpy.zeros(batch_size)

		idxs = numpy.random.randint(len(data), size=batch_size)

		for i, idx in enumerate(idxs):
			celltype_idxs[i] = indices[0, idx]
			assay_idxs[i]    = indices[1, idx]
			value[i]         = tracks[idx][genomic_25bp_idxs[i]]

		d = {
			'celltype_input': celltype_idxs, 
			'assay_input': assay_idxs, 
			'genome_25bp_input': genomic_25bp_idxs, 
			'genome_250bp_input': genomic_250bp_idxs,
			'genome_5kbp_input': genomic_5kbp_idxs
		}

		yield d, value

		start += batch_size

def sequential_data_generator(celltypes, assays, data, n_positions, batch_size):
	"""Yield batches that scan the genomic bins in order.

	Successive batches cover consecutive runs of `batch_size` bins, starting at
	bin 0 and wrapping around at `n_positions`. The experiment for each bin is
	drawn uniformly at random from the keys of `data`. This is the generator
	that `Avocado.fit` uses by default (`sampling='sequential'`), and the only
	training generator that `fit_celltypes` and `fit_assays` use.

	Parameters
	----------
	celltypes : list
		The cell types in the model, in the order of the cell type embedding.
		Each example's cell type is encoded as its index in this list.

	assays : list
		The assays in the model, in the order of the assay embedding. Each
		example's assay is encoded as its index in this list.

	data : dict
		A dictionary where the keys are (celltype, assay) tuples and the values
		are tracks, either numpy arrays in memory or memory maps, each with at
		least `n_positions` entries.

	n_positions : int
		The number of 25 bp genomic bins to draw from.

	batch_size : int
		The number of examples in each batch.

	Yields
	------
	X : dict
		A dictionary mapping the five input layer names ('celltype_input',
		'assay_input', 'genome_25bp_input', 'genome_250bp_input' and
		'genome_5kbp_input') to integer arrays of shape (batch_size,). The
		250 bp and 5 kbp indexes are the 25 bp index divided by 10 and 200.

	y : numpy.ndarray, shape=(batch_size,)
		The signal value of each example's track at its 25 bp bin, as float64.
	"""

	start = 0

	indices = numpy.array([[celltypes.index(celltype) for celltype, _ in data.keys()],
		                   [assays.index(assay) for _, assay in data.keys()]])

	tracks = list(data.values())

	while True:
		celltype_idxs      = numpy.zeros(batch_size, dtype='int32')
		assay_idxs         = numpy.zeros(batch_size, dtype='int32')
		genomic_25bp_idxs  = numpy.arange(start, start+batch_size) % n_positions
		genomic_250bp_idxs = genomic_25bp_idxs // 10
		genomic_5kbp_idxs  = genomic_25bp_idxs // 200
		value              = numpy.zeros(batch_size)

		idxs = numpy.random.randint(len(data), size=batch_size)

		for i, idx in enumerate(idxs):
			celltype_idxs[i] = indices[0, idx]
			assay_idxs[i]    = indices[1, idx]
			value[i]         = tracks[idx][genomic_25bp_idxs[i]]

		d = {
			'celltype_input': celltype_idxs, 
			'assay_input': assay_idxs, 
			'genome_25bp_input': genomic_25bp_idxs, 
			'genome_250bp_input': genomic_250bp_idxs,
			'genome_5kbp_input': genomic_5kbp_idxs
		}

		yield d, value

		start += batch_size
		
