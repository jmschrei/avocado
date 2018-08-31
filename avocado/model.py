# models.py
# Contact: Jacob Schreiber <jmschr@cs.washington.edu>
#          William Noble   <wnoble@uw.edu>

"""
Avocado is deep tensor factorization model for learning a latent representation
of the human epigenome. This file has functions for building a deep tensor
factorization model.
"""

from .io import sequential_data_generator
from .io import data_generator

import numpy

try:
	import keras
	from keras.layers import Input, Embedding, Dense
	from keras.layers import Multiply, Dot, Flatten, concatenate
	from keras.models import Model
	from keras.optimizers import Adam
except:
	Input, Embedding, Dense = object, object, object
	Multiply, Dot, Flatten = object, object, object
	Model, Adam = object, object

def build_model(n_celltypes, n_celltype_factors, n_assays, n_assay_factors,
	n_genomic_positions, n_25bp_factors, n_250bp_factors, n_5kbp_factors,
	n_layers, n_nodes):
	"""This function builds a multi-scale deep tensor factorization model."""

	celltype_input = Input(shape=(1,), name="celltype")
	celltype_embedding = Embedding(n_celltypes, n_celltype_factors, input_length=1)(celltype_input)
	celltype = Flatten()(celltype_embedding)

	assay_input = Input(shape=(1,), name="assay")
	assay_embedding = Embedding(n_assays, n_assay_factors, input_length=1)
	assay = Flatten()(assay_embedding(assay_input))

	genome_25bp_input = Input(shape=(1,), name="genome_25bp")
	genome_25bp_embedding = Embedding(n_genomic_positions, n_25bp_factors, input_length=1)
	genome_25bp = Flatten()(genome_25bp_embedding(genome_25bp_input))

	genome_250bp_input = Input(shape=(1,), name="genome_250bp")
	genome_250bp_embedding = Embedding((n_genomic_positions / 10) + 1, n_250bp_factors, input_length=1)
	genome_250bp = Flatten()(genome_250bp_embedding(genome_250bp_input))

	genome_5kbp_input = Input(shape=(1,), name="genome_5kbp")
	genome_5kbp_embedding = Embedding((n_genomic_positions / 200) + 1, n_5kbp_factors, input_length=1)
	genome_5kbp = Flatten()(genome_5kbp_embedding(genome_5kbp_input))

	layers = [celltype, assay, genome_25bp, genome_250bp, genome_5kbp]
	inputs = (celltype_input, assay_input, genome_25bp_input, genome_250bp_input, genome_5kbp_input)

	x = concatenate(layers)
	for i in range(n_layers):
		x = Dense(n_nodes, activation='relu')(x)

	y = Dense(1)(x)

	model = Model(inputs=inputs, outputs=y)
	model.compile(optimizer='adam', loss='mse', metrics=['mse'])
	return model

class Avocado(object):
	def __init__(self, celltypes, assays, n_celltype_factors=32, 
		n_assay_factors=256, n_genomic_positions=1126469,
		n_25bp_factors=25, n_250bp_factors=40, n_5kbp_factors=45, n_layers=2,
		n_nodes=2048, batch_size=40000):

		self.celltypes = celltypes
		self.assays = assays

		self.n_celltypes = len(celltypes)
		self.n_assays = len(assays)

		self.n_genomic_positions = n_genomic_positions
		self.batch_size = batch_size

		self.model = build_model(n_celltypes=self.n_celltypes,
								 n_celltype_factors=n_celltype_factors,
								 n_assays=self.n_assays,
								 n_assay_factors=n_assay_factors,
								 n_genomic_positions=n_genomic_positions,
								 n_25bp_factors=n_25bp_factors,
								 n_250bp_factors=n_250bp_factors,
								 n_5kbp_factors=n_5kbp_factors,
								 n_layers=n_layers,
								 n_nodes=n_nodes)

	def summary(self):
		"""A wrapper method for the keras summary method."""

		self.model.summary()

	def fit(self, X_train, X_valid=None, n_epochs=200, epoch_size=120):
		"""Fit the model to the given epigenomic tracks.

		Pass in a dictionary of training data and an optional dictionary of
		validation data. The keys to this dictionary are a tuple of the format
		(celltype, assay) and the values are the corresponding track in the
		form of a numpy array. The tracks can either be in the form of an array
		that is in memory or as a memory map.

		Parameters
		----------
		X_train : dict
			A dictionary of training data values, where the keys are a tuple of
			(celltype, assay) and the values are a track.

		X_valid : dict or None, optional
			A dictionary of validation data values that are used to calculate
			validation set MSE during the training process. If None, validation
			set statistics are not calculated during the training process.
			Default is None.

		n_epochs : int, optional
			The number of epochs to train on before ending training. Default is 120.

		epoch_size : int, optional
			The number of batches per epoch. Default is 200.
		"""

		X_train_gen = sequential_data_generator(self.celltypes, self.assays, 
			X_train, self.n_genomic_positions, self.batch_size)

		if X_valid is not None:
			X_valid_gen = data_generator(self.celltypes, self.assays, 
				X_valid, self.n_genomic_positions, self.batch_size)
			self.model.fit_generator(X_train_gen, epoch_size, n_epochs, workers=1, 
				pickle_safe=True, validation_data=X_valid_gen, validation_steps=30)
		else:
			self.model.fit_generator(X_train_gen, epoch_size, n_epochs, workers=1, pickle_safe=True)

	def predict(self, celltype, assay):
		celltype_idx = self.celltypes.index(celltype)
		assay_idx = self.assays.index(assay)

		celltype_idxs = numpy.ones(self.n_genomic_positions) * celltype_idx
		assay_idxs = numpy.ones(self.n_genomic_positions) * assay_idx

		genomic_25bp_idxs  = numpy.arange(self.n_genomic_positions)
		genomic_250bp_idxs = numpy.arange(self.n_genomic_positions) / 10
		genomic_5kbp_idxs  = numpy.arange(self.n_genomic_positions) / 200

		X = {'celltype' : celltype_idxs, 'assay' : assay_idxs, 
			'genome_25bp' : genomic_25bp_idxs, 'genome_250bp' : genomic_250bp_idxs, 
			'genome_5kbp' : genomic_5kbp_idxs}
		y = self.model.predict(X, batch_size=self.batch_size)[:,0]
		return y

	def get_params(self):
		params = []
		for layer in model.layers:
			params.append(layers.get_weghts()[0])

	def save(self, name="avocado"):
		"""derp"""

		self.model.save("{}.h5".format(name))