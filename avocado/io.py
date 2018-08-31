import numpy

def data_generator(celltypes, assays, data, n_positions, batch_size):
	while True:
		celltype_idxs      = numpy.zeros(batch_size, dtype='int32')
		assay_idxs         = numpy.zeros(batch_size, dtype='int32')
		genomic_25bp_idxs  = numpy.random.randint(n_positions, size=batch_size)
		genomic_250bp_idxs = genomic_25bp_idxs / 10
		genomic_5kbp_idxs  = genomic_25bp_idxs / 200
		value              = numpy.zeros(batch_size)

		keys = data.keys()
		idxs = numpy.random.randint(len(data), size=batch_size)

		for i, idx in enumerate(idxs):
			celltype, assay = keys[idx]
			track = data[(celltype, assay)]

			celltype_idxs[i] = celltypes.index(celltype)
			assay_idxs[i]    = assays.index(assay)
			value[i]         = track[genomic_25bp_idxs[i]]

		yield {'celltype' : celltype_idxs, 'assay' : assay_idxs, 
			'genome_25bp' : genomic_25bp_idxs, 'genome_250bp' : genomic_250bp_idxs,
			'genome_5kbp' : genomic_5kbp_idxs}, value

def sequential_data_generator(celltypes, assays, data, n_positions, batch_size):
	start = 0
	while True:
		celltype_idxs      = numpy.zeros(batch_size, dtype='int32')
		assay_idxs         = numpy.zeros(batch_size, dtype='int32')
		genomic_25bp_idxs  = numpy.arange(start, start+batch_size) % n_positions
		genomic_250bp_idxs = genomic_25bp_idxs / 10
		genomic_5kbp_idxs  = genomic_25bp_idxs / 200
		value              = numpy.zeros(batch_size)

		keys = data.keys()
		idxs = numpy.random.randint(len(data), size=batch_size)

		for i, idx in enumerate(idxs):
			celltype, assay = keys[idx]
			track = data[(celltype, assay)]

			celltype_idxs[i] = celltypes.index(celltype)
			assay_idxs[i]    = assays.index(assay)
			value[i]         = track[genomic_25bp_idxs[i]]

		yield {'celltype' : celltype_idxs, 'assay' : assay_idxs, 
			'genome_25bp' : genomic_25bp_idxs, 'genome_250bp' : genomic_250bp_idxs,
			'genome_5kbp' : genomic_5kbp_idxs}, value

		start += batch_size