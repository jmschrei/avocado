import numpy

def data_generator(celltypes, assays, data, n_positions, batch_size):
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
		
