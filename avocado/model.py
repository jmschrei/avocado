# model.py
# Contact: Jacob Schreiber <jmschr@cs.washington.edu>
#          William Noble   <wnoble@uw.edu>

"""
Avocado is deep tensor factorization model for learning a latent representation
of the human epigenome. This file has functions for building a deep tensor
factorization model.
"""

from .io import data_generator
from .io import permuted_data_generator
from .io import sequential_data_generator


import json
import numpy
import keras

from keras.layers import Input, Embedding, Dense
from keras.layers import Multiply, Dot, Flatten, concatenate
from keras.models import Model
from keras.optimizers import Adam

def build_model(n_celltypes, n_celltype_factors, n_assays, n_assay_factors,
	n_genomic_positions, n_25bp_factors, n_250bp_factors, n_5kbp_factors,
	n_layers, n_nodes, freeze_celltypes=False, freeze_assays=False,
	freeze_genome_25bp=False, freeze_genome_250bp=False, 
	freeze_genome_5kbp=False, freeze_network=False):
	"""Build and compile the Keras model behind an Avocado object.

	The model has five integer inputs: the cell type, the assay, and the
	genomic position at 25 bp, 250 bp and 5 kbp resolution. Each input is
	looked up in its own embedding layer, the five embeddings are concatenated,
	and the result passes through `n_layers` dense layers of `n_nodes` ReLU
	units and a final linear unit that predicts the signal value. The model is
	compiled with the Adam optimizer and the mean squared error loss.

	The 25 bp embedding has `n_genomic_positions` rows, the 250 bp embedding
	has `int(n_genomic_positions / 10) + 1` rows and the 5 kbp embedding has
	`int(n_genomic_positions / 200) + 1` rows. The data generators and
	`Avocado.predict` index the coarser embeddings with the 25 bp index
	divided by 10 and by 200.

	Each `freeze_*` argument sets `trainable = False` on its layers before the
	model is compiled, so `fit` does not update those weights.
	`Avocado.fit_celltypes` and `Avocado.fit_assays` use them to train one
	embedding while holding the rest of the model fixed.

	Parameters
	----------
	n_celltypes : int
		The number of cell types, which is the number of rows in the cell type
		embedding.

	n_celltype_factors : int
		The number of factors used to represent each cell type.

	n_assays : int
		The number of assays, which is the number of rows in the assay
		embedding.

	n_assay_factors : int
		The number of factors used to represent each assay.

	n_genomic_positions : int
		The number of 25 bp genomic bins, which is the number of rows in the
		25 bp genome embedding.

	n_25bp_factors : int
		The number of factors used to represent each 25 bp bin.

	n_250bp_factors : int
		The number of factors used to represent each 250 bp bin.

	n_5kbp_factors : int
		The number of factors used to represent each 5 kbp bin.

	n_layers : int
		The number of hidden dense layers.

	n_nodes : int
		The number of ReLU units in each hidden dense layer.

	freeze_celltypes : bool, optional
		Whether to freeze the cell type embedding. Default is False.

	freeze_assays : bool, optional
		Whether to freeze the assay embedding. Default is False.

	freeze_genome_25bp : bool, optional
		Whether to freeze the 25 bp genome embedding. Default is False.

	freeze_genome_250bp : bool, optional
		Whether to freeze the 250 bp genome embedding. Default is False.

	freeze_genome_5kbp : bool, optional
		Whether to freeze the 5 kbp genome embedding. Default is False.

	freeze_network : bool, optional
		Whether to freeze the dense layers, including the output layer.
		Default is False.

	Returns
	-------
	model : keras.models.Model
		The compiled Keras model.
	"""

	celltype_input = Input(shape=(1,), name="celltype_input")
	celltype_embedding = Embedding(n_celltypes, n_celltype_factors, 
		input_length=1, name="celltype_embedding")
	celltype_embedding.trainable = not freeze_celltypes
	celltype = Flatten()(celltype_embedding(celltype_input))

	assay_input = Input(shape=(1,), name="assay_input")
	assay_embedding = Embedding(n_assays, n_assay_factors, 
		input_length=1, name="assay_embedding")
	assay_embedding.trainable = not freeze_assays
	assay = Flatten()(assay_embedding(assay_input))

	genome_25bp_input = Input(shape=(1,), name="genome_25bp_input")
	genome_25bp_embedding = Embedding(n_genomic_positions, n_25bp_factors, 
		input_length=1, name="genome_25bp_embedding")
	genome_25bp_embedding.trainable = not freeze_genome_25bp
	genome_25bp = Flatten()(genome_25bp_embedding(genome_25bp_input))

	genome_250bp_input = Input(shape=(1,), name="genome_250bp_input")
	genome_250bp_embedding = Embedding(int(n_genomic_positions / 10) + 1,
		n_250bp_factors, input_length=1, name="genome_250bp_embedding")
	genome_250bp_embedding.trainable = not freeze_genome_250bp
	genome_250bp = Flatten()(genome_250bp_embedding(genome_250bp_input))

	genome_5kbp_input = Input(shape=(1,), name="genome_5kbp_input")
	genome_5kbp_embedding = Embedding(int(n_genomic_positions / 200) + 1, 
		n_5kbp_factors, input_length=1, name="genome_5kbp_embedding")
	genome_5kbp_embedding.trainable = not freeze_genome_5kbp
	genome_5kbp = Flatten()(genome_5kbp_embedding(genome_5kbp_input))

	layers = [celltype, assay, genome_25bp, genome_250bp, genome_5kbp]
	inputs = (celltype_input, assay_input, genome_25bp_input, 
		genome_250bp_input, genome_5kbp_input)

	x = concatenate(layers)
	for i in range(n_layers):
		layer = Dense(n_nodes, activation='relu', name="dense_{}".format(i))
		layer.trainable = not freeze_network
		x = layer(x)

	layer = Dense(1, name="y_pred")
	layer.trainable = not freeze_network
	y = layer(x)

	model = Model(inputs=inputs, outputs=y)
	model.compile(optimizer='adam', loss='mse', metrics=['mse'])
	return model

class Avocado(object):
	"""An Avocado multi-scale deep tensor factorization model.

	The Avocado model is a multi-scale deep tensor factorization model. It is
	multi-scale because it represents the genome axis using three different
	resolutions---25 bp, 250 bp and 5 kbp. It is deep because it replaces the
	dot product component of most linear factorization approaches with a deep
	neural network. The tensor factors and the neural network weights are
	trained jointly to impute the values in the tensor that it is provided.

	In this case Avocado is trained on epigenomic data whose dimensions are
	human cell type, epigenomic assay, and genomic coordinate. The trained
	model can impute epigenomic assays that have not yet been performed, and
	the learned factor values can themselves be used to represent genomic
	positions more compactly than the full set of epigenomic measurements
	could.

	The default parameters are those used in the manuscript entitled 
	"Multi-scale deep tensor factorization learns a latent representation
	of the human epigenome". 

	Parameters
	----------
	celltypes : list
		The list of cell type names that will be modeled

	assays : list
		The list of assays that will be modeled

	n_celltype_factors : int, optional
		The number of factors to use to represent each cell type. Default is 32.

	n_assay_factors : int, optional
		The number of factors to use to represent each assay. Default is 256.

	n_genomic_positions : int, optional
		The number of genomic positions to model. This is typically either
		the size of the pilot regions when performing initial training or
		the size of the chromosome when fitting the genomic latent factors.
		Default is 1126469, the size of the pilot regions in chr1-22.

	n_25bp_factors : int, optional
		The number of factors to use to represent the genome at 25 bp
		resolution. Default is 25.

	n_250bp_factors : int, optional
		The number of factors to use to represent the genome at 250 bp
		resolution. Default is 40.

	n_5kbp_factors : int, optional
		The number of factors to use to represent the genome at 5 kbp
		resolution. Default is 45.

	n_layers : int, optional
		The number of hidden layers in the neural model. Default is 2.

	n_nodes : int, optional
		The number of nodes per layer. Default is 2048.

	batch_size : int, optional
		The number of examples in each batch, used both in training and in
		`predict`. Default is 40000.

	freeze_celltypes : bool, optional
		Whether to freeze the training of the cell type embedding. Default
		is False.

	freeze_assays : bool, optional
		Whether to freeze the training of the assay embeddings. Default
		is False.

	freeze_genome_25bp : bool, optional
		Whether to freeze the training of the 25 bp genome factors. Default
		is False.

	freeze_genome_250bp : bool, optional
		Whether to freeze the training of the 250 bp genome factors. Default
		is False.

	freeze_genome_5kbp : bool, optional
		Whether to freeze the training of the 5 kbp genome factors. Default
		is False.

	freeze_network : bool, optional
		Whether to freeze the training of the neural network. Default
		is False.

	Example
	-------
	>>> import numpy, itertools
	>>> from avocado import Avocado
	>>>
	>>> celltypes = ['E003', 'E017', 'E065', 'E116', 'E117']
	>>> assays = ['H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K4me1']
	>>> 
	>>> data = {}
	>>> for celltype, assay in itertools.product(celltypes, assays):
	>>>    filename = 'data/{}.{}.pilot.arcsinh.npz'.format(celltype, assay)
	>>>    data[(celltype, assay)] = numpy.load(filename)['arr_0']
	>>>
	>>> model = Avocado(celltypes, assays)
	>>> model.fit(data)
	>>>
	>>> track = model.predict("E065", "H3K27me3")
	"""

	def __init__(self, celltypes, assays, n_celltype_factors=32, 
		n_assay_factors=256, n_genomic_positions=1126469,
		n_25bp_factors=25, n_250bp_factors=40, n_5kbp_factors=45, n_layers=2,
		n_nodes=2048, batch_size=40000, freeze_celltypes=False, 
		freeze_assays=False, freeze_genome_25bp=False, freeze_genome_250bp=False,
		freeze_genome_5kbp=False, freeze_network=False):

		self.celltypes = list(celltypes)
		self.assays = list(assays)
		self.experiments = []

		self.n_celltypes = len(celltypes)
		self.n_assays = len(assays)

		self.batch_size = batch_size

		self.n_celltype_factors = n_celltype_factors
		self.n_celltype_factors = n_celltype_factors
		self.n_assay_factors = n_assay_factors
		self.n_genomic_positions = n_genomic_positions
		self.n_25bp_factors = n_25bp_factors
		self.n_250bp_factors = n_250bp_factors
		self.n_5kbp_factors = n_5kbp_factors
		self.n_layers = n_layers
		self.n_nodes = n_nodes

		self.freeze_celltypes = freeze_celltypes
		self.freeze_assays = freeze_assays
		self.freeze_genome_25bp = freeze_genome_25bp
		self.freeze_genome_250bp = freeze_genome_250bp
		self.freeze_genome_5kbp = freeze_genome_5kbp
		self.freeze_network = freeze_network

		self.model = build_model(n_celltypes=self.n_celltypes,
								 n_celltype_factors=n_celltype_factors,
								 n_assays=self.n_assays,
								 n_assay_factors=n_assay_factors,
								 n_genomic_positions=n_genomic_positions,
								 n_25bp_factors=n_25bp_factors,
								 n_250bp_factors=n_250bp_factors,
								 n_5kbp_factors=n_5kbp_factors,
								 n_layers=n_layers,
								 n_nodes=n_nodes,
								 freeze_celltypes=freeze_celltypes,
								 freeze_assays=freeze_assays,
								 freeze_genome_25bp=freeze_genome_25bp,
								 freeze_genome_250bp=freeze_genome_250bp,
								 freeze_genome_5kbp=freeze_genome_5kbp,
								 freeze_network=freeze_network)

	@property
	def celltype_embedding(self):
		"""Returns the learned cell type embedding as a numpy array.

		Parameters
		----------
		None

		Returns
		-------
		celltype_embedding : numpy.ndarray, shape=(n_celltypes, n_factors)
			The learned embedding corresponding to the input name 
			'celltype_embedding'. The cell types are ordered according to the
			order defined in self.celltypes.
		"""

		for layer in self.model.layers:
			if layer.name == 'celltype_embedding':
				return layer.get_weights()[0]

		raise ValueError("No layer in model named 'celltype_embedding'.")

	@property
	def assay_embedding(self):
		"""Returns the learned assay embedding as a numpy array.

		Parameters
		----------
		None

		Returns
		-------
		assay_embedding : numpy.ndarray, shape=(n_assays, n_factors)
			The learned embedding corresponding to the input name
			'assay_embedding'. The assays are ordered according to the order 
			defined in self.assays.
		"""

		for layer in self.model.layers:
			if layer.name == 'assay_embedding':
				return layer.get_weights()[0]

		raise ValueError("No layer in model named 'assay_embedding'.")


	@property
	def genome_embedding(self):
		"""Returns the learned genomic embedding as a numpy array.

		This function will concatenate together the three resolutions of
		genomic factors, such that the first columns correspond to the
		25 bp factors, the next columns correspond to the 250 bp factors,
		and the final columns correspond to the 5 kbp factors. The factors
		that span more than 25 bp are repeated across successive positions:
		row i holds row i // 10 of the 250 bp embedding and row i // 200 of
		the 5 kbp embedding.

		The array is float64 with one row per 25 bp bin, so for a whole
		chromosome it can take several gigabytes of memory.

		Parameters
		----------
		None

		Returns
		-------
		genome_embedding : numpy.ndarray, shape=(n_genomic_positions, 
			n_25bp_factors + n_250bp_factors + n_5kbp_factors)
			The learned embedding corresponding to the input names
			genome_25bp_embedding, genome_250bp_embedding, and 
			genome_5kbp_embedding.
		"""

		n_25bp = self.n_25bp_factors
		n_250bp = self.n_250bp_factors
		n_5kbp = self.n_5kbp_factors

		genome_embedding = numpy.empty((self.n_genomic_positions, 
			n_25bp + n_250bp + n_5kbp))

		for layer in self.model.layers:
			if layer.name == 'genome_25bp_embedding':
				genome_25bp_embedding = layer.get_weights()[0]
			elif layer.name == 'genome_250bp_embedding':
				genome_250bp_embedding = layer.get_weights()[0]
			elif layer.name == 'genome_5kbp_embedding':
				genome_5kbp_embedding = layer.get_weights()[0]

		n1 = n_25bp
		n2 = n_25bp + n_250bp

		for i in range(self.n_genomic_positions):
			genome_embedding[i, :n1] = genome_25bp_embedding[i]
			genome_embedding[i, n1:n2] = genome_250bp_embedding[i // 10]
			genome_embedding[i, n2:] = genome_5kbp_embedding[i // 200]

		return genome_embedding

	def summary(self):
		"""A wrapper method for the keras summary method."""

		self.model.summary()

	def fit(self, X_train, X_valid=None, n_epochs=200, epoch_size=120,
		verbose=1, callbacks=None, sampling='sequential', input_generator=None, 
		**kwargs):
		"""Fit the model to the given epigenomic tracks.

		Pass in a dictionary of training data and an optional dictionary of
		validation data. The keys to this dictionary are a tuple of the format
		(celltype, assay) and the values are the corresponding track in the
		form of a numpy array. The tracks can either be in the form of an array
		that is in memory or as a memory map.

		Each call continues training from the current weights. Training runs
		through Keras' `fit_generator` with `epoch_size` steps per epoch, and
		after training `self.experiments` is replaced by the keys of
		`X_train`.

		Parameters
		----------
		X_train : dict
			A dictionary of training data values, where the keys are a tuple of
			(celltype, assay) and the values are a track. Every cell type and
			assay must be in the model and every track must have
			`n_genomic_positions` entries, or a ValueError is raised.

		X_valid : dict or None, optional
			A dictionary of validation data values that are used to calculate
			validation set MSE during the training process, from 30 batches at
			the end of each epoch. If None, validation set statistics are not
			calculated during the training process. On Python 3, passing a
			dictionary raises a TypeError at the end of the first epoch; see
			`avocado.io.data_generator`. Default is None.

		n_epochs : int, optional
			The number of epochs to train on before ending training. Default is 200.

		epoch_size : int, optional
			The number of batches per epoch. Default is 120.

		verbose: int, optional
			The verbosity level of training. Must be one of 0, 1, or 2, where 0
			means silent, 1 means progress bar, and 2 means use only one line
			per epoch. Default is 1.

		callbacks : list or None, optional
			A list of keras callback instances to be called during training.

		sampling : str, optional
			The sampling strategy to use for the generators. Must be one of the
			following:

				'sequential' : Sequentially scans through the genome indexes,
					selecting a cell type and assay randomly at each position
				'permuted' : Sequentially scans through a permuted version
					of the genome indexes, such that each epoch sees every
					genomic index once, but each batch sees nearly random 
					indexes
				'random' : Uses the same generator as 'permuted'.

			Any other value raises an UnboundLocalError. Ignored when
			`input_generator` is given. Default is 'sequential'.

		input_generator : generator or None, optional
			A custom data generator object to be used in the place of the
			default generator. It must yield the same (X, y) batches as the
			generators in `avocado.io`. This will only change the training
			generator, not the validation generator. Default is None.

		**kwargs : optional
			Any other keyword arguments to be passed into the `fit_generator`
			method.

		Returns
		-------
		history : keras.callbacks.History
			The keras history object that records training loss values and
			metric values in its `history` attribute.
		"""

		if not isinstance(X_train, dict):
			raise ValueError("X_train must be a dictionary where the keys" \
				" are (celltype, assay) tuples and the values are the track" \
				" corresponding to that pair.")

		if X_valid is not None and not isinstance(X_valid, dict):
			raise ValueError("X_valid must be a dictionary where the keys" \
				" are (celltype, assay) tuples and the values are the track" \
				" corresponding to that pair.")	

		for (celltype, assay), track in X_train.items():
			if celltype not in self.celltypes:
				raise ValueError("Celltype {} appears in the training data " \
					"but not in the list of cell types provided to the " \
					"model.".format(celltype))

			if assay not in self.assays:
				raise ValueError("Assay {} appears in the training data " \
					"but not in the list of assays provided to the " \
					"model.".format(assay))

			if len(track) != self.n_genomic_positions:
				raise ValueError("The track corresponding to {} {} is of " \
					"size {} while the model encodes {} genomic " \
					"positions".format(celltype, assay, len(track), 
						self.n_genomic_positions))

		if X_valid is not None:
			for (celltype, assay), track in X_valid.items():
				if celltype not in self.celltypes:
					raise ValueError("Celltype {} appears in the validation " \
						"data but not in the list of cell types provided to " \
						"the model.".format(celltype))

				if assay not in self.assays:
					raise ValueError("Assay {} appears in the validation " \
						"data but not in the list of assays provided to the " \
						"model.".format(assay))

				if len(track) != self.n_genomic_positions:
					raise ValueError("The track corresponding to {} {} is of " \
						"size {} while the model encodes {} genomic " \
						"positions".format(celltype, assay, len(track), 
							self.n_genomic_positions))

		if input_generator is not None:
			X_train_gen = input_generator
		elif sampling == 'sequential':
			X_train_gen = sequential_data_generator(self.celltypes, 
				self.assays, X_train, self.n_genomic_positions, 
				self.batch_size)
		elif sampling == 'permuted':
			X_train_gen = permuted_data_generator(self.celltypes, 
				self.assays, X_train, self.n_genomic_positions, 
				self.batch_size)
		elif sampling == 'random':
			X_train_gen = permuted_data_generator(self.celltypes, 
				self.assays, X_train, self.n_genomic_positions, 
				self.batch_size)			

		if X_valid is not None:
			X_valid_gen = data_generator(self.celltypes, self.assays, 
				X_valid, self.n_genomic_positions, self.batch_size)

			history = self.model.fit_generator(X_train_gen, epoch_size, n_epochs, 
				workers=1,  validation_data=X_valid_gen, 
				validation_steps=30, verbose=verbose, callbacks=callbacks, 
				**kwargs)
		else:
			history = self.model.fit_generator(X_train_gen, epoch_size, n_epochs, 
				workers=1,  verbose=verbose, 
				callbacks=callbacks, **kwargs)

		self.experiments = list(X_train.keys())
		return history

	def fit_celltypes(self, X_train, X_valid=None, n_epochs=200, epoch_size=120,
		verbose=1, callbacks=None, **kwargs):
		"""Add a new cell type(s) to an otherwise frozen model.

		This method will add a new cell type to the cell type embedding after
		freezing all of the other parameters in the model, including weights
		and the other cell type positions. It builds a temporary model whose
		cell type embedding holds only the new cell types, copies every other
		layer from this model and freezes it, and trains the new embedding
		using the sequential generator. It then replaces `self.model` with a
		newly built and compiled model whose cell type embedding is the old
		embedding with the new rows appended, and appends the new cell types
		to `self.celltypes` in sorted order. Because the replacement model is
		newly compiled, the optimizer state from earlier calls to `fit` is not
		kept.

		Pass in a dictionary of training data and an optional dictionary of
		validation data. The keys to this dictionary are a tuple of the format
		(celltype, assay) and the values are the corresponding track in the
		form of a numpy array. The tracks can either be in the form of an array
		that is in memory or as a memory map. The celltypes provided should not
		appear in the model.celltypes attribute but the assays should exclusively
		appear in the model.assays attribute.

		`self.experiments` is not updated with the new experiments: the
		combined list is assigned to the `experiments` attribute of the new
		Keras model instead, so `save` writes the experiments from before this
		call.

		Parameters
		----------
		X_train : dict
			A dictionary of training data values, where the keys are a tuple of
			(celltype, assay) and the values are a track.

		X_valid : dict or None, optional
			A dictionary of validation data values that are used to calculate
			validation set MSE during the training process. If None, validation
			set statistics are not calculated during the training process. On
			Python 3, passing a dictionary raises a TypeError at the end of the
			first epoch; see `avocado.io.data_generator`. Default is None.

		n_epochs : int, optional
			The number of epochs to train on before ending training. Default is 200.

		epoch_size : int, optional
			The number of batches per epoch. Default is 120.

		verbose: int, optional
			The verbosity level of training. Must be one of 0, 1, or 2, where 0
			means silent, 1 means progress bar, and 2 means use only one line
			per epoch. Default is 1.

		callbacks : list or None, optional
			A list of keras callback instances to be called during training. 

		**kwargs : optional
			Any other keyword arguments to be passed into the `fit_generator`
			method.

		Returns
		-------
		history : keras.callbacks.History
			The keras history object of the temporary model, which records
			training loss values and metric values in its `history` attribute.
		"""

		if not isinstance(X_train, dict):
			raise ValueError("X_train must be a dictionary where the keys" \
				" are (celltype, assay) tuples and the values are the track" \
				" corresponding to that pair.")

		if X_valid is not None and not isinstance(X_valid, dict):
			raise ValueError("X_valid must be a dictionary where the keys" \
				" are (celltype, assay) tuples and the values are the track" \
				" corresponding to that pair.")	

		for (celltype, assay), track in X_train.items():
			if celltype in self.celltypes:
				raise ValueError("Celltype {} appears in the training data " \
					"and also in the list of cell types already in the " \
					"model.".format(celltype))

			if assay not in self.assays:
				raise ValueError("Assay {} appears in the training data " \
					"but not in the list of assays provided to the " \
					"model.".format(assay))

			if len(track) != self.n_genomic_positions:
				raise ValueError("The track corresponding to {} {} is of " \
					"size {} while the model encodes {} genomic " \
					"positions".format(celltype, assay, len(track), 
						self.n_genomic_positions))

		if X_valid is not None:
			for (celltype, assay), track in X_valid.items():
				if celltype in self.celltypes:
					raise ValueError("Celltype {} appears in the validation " \
						"data and also in the list of cell types already in " \
						"the model.".format(celltype))

				if assay not in self.assays:
					raise ValueError("Assay {} appears in the training data " \
						"but not in the list of assays provided to the " \
						"model.".format(assay))

				if len(track) != self.n_genomic_positions:
					raise ValueError("The track corresponding to {} {} is of " \
						"size {} while the model encodes {} genomic " \
						"positions".format(celltype, assay, len(track), 
							self.n_genomic_positions))

		new_celltypes = list(numpy.unique([ct for ct, _ in X_train.keys()]))

		model = build_model(n_celltypes=len(new_celltypes),
							n_celltype_factors=self.n_celltype_factors,
							n_assays=self.n_assays,
							n_assay_factors=self.n_assay_factors,
							n_genomic_positions=self.n_genomic_positions,
							n_25bp_factors=self.n_25bp_factors,
							n_250bp_factors=self.n_250bp_factors,
							n_5kbp_factors=self.n_5kbp_factors,
							n_layers=self.n_layers,
							n_nodes=self.n_nodes,
							freeze_celltypes=False,
							freeze_assays=True,
							freeze_genome_25bp=True,
							freeze_genome_250bp=True,
							freeze_genome_5kbp=True,
							freeze_network=True)

		for old_layer, new_layer in zip(self.model.layers, model.layers):
			if 'input' in old_layer.name:
				continue
			if old_layer.name == 'celltype_embedding':
				continue

			new_layer.set_weights(old_layer.get_weights())


		X_train_gen = sequential_data_generator(new_celltypes, self.assays, 
			X_train, self.n_genomic_positions, self.batch_size)

		if X_valid is not None:
			X_valid_gen = data_generator(new_celltypes, self.assays, 
				X_valid, self.n_genomic_positions, self.batch_size)

			history = model.fit_generator(X_train_gen, epoch_size, n_epochs, 
				workers=1,  validation_data=X_valid_gen, 
				validation_steps=30, verbose=verbose, callbacks=callbacks, 
				**kwargs)
		else:
			history = model.fit_generator(X_train_gen, epoch_size, n_epochs, 
				workers=1,  verbose=verbose, 
				callbacks=callbacks, **kwargs)

		for layer in self.model.layers:
			if layer.name == 'celltype_embedding':
				celltype_embedding = layer.get_weights()[0]
				break

		for layer in model.layers:
			if layer.name == 'celltype_embedding':
				new_celltype_embedding = layer.get_weights()[0]
				break

		celltype_embedding = numpy.concatenate([celltype_embedding, 
			new_celltype_embedding]) 

		self.celltypes.extend(new_celltypes)
		self.n_celltypes = len(self.celltypes)

		model = build_model(n_celltypes=self.n_celltypes,
							n_celltype_factors=self.n_celltype_factors,
							n_assays=self.n_assays,
							n_assay_factors=self.n_assay_factors,
							n_genomic_positions=self.n_genomic_positions,
							n_25bp_factors=self.n_25bp_factors,
							n_250bp_factors=self.n_250bp_factors,
							n_5kbp_factors=self.n_5kbp_factors,
							n_layers=self.n_layers,
							n_nodes=self.n_nodes,
							freeze_celltypes=self.freeze_celltypes,
							freeze_assays=self.freeze_assays,
							freeze_genome_25bp=self.freeze_genome_25bp,
							freeze_genome_250bp=self.freeze_genome_250bp,
							freeze_genome_5kbp=self.freeze_genome_5kbp,
							freeze_network=self.freeze_network)

		for old_layer, new_layer in zip(self.model.layers, model.layers):
			if 'input' in old_layer.name:
				continue
			if old_layer.name == 'celltype_embedding':
				new_layer.set_weights([celltype_embedding])
			else:
				new_layer.set_weights(old_layer.get_weights())

		model.experiments = self.experiments + list(X_train.keys())
		self.model = model
		return history

	def fit_assays(self, X_train, X_valid=None, n_epochs=200, epoch_size=120,
		verbose=1, callbacks=None, **kwargs):
		"""Add a new assay(s) to an otherwise frozen model.

		This method will add a new assay to the assay embedding after
		freezing all of the other parameters in the model, including weights
		and the other assay positions. It builds a temporary model whose
		assay embedding holds only the new assays, copies every other layer
		from this model and freezes it, and trains the new embedding using
		the sequential generator. It then replaces `self.model` with a newly
		built and compiled model whose assay embedding is the old embedding
		with the new rows appended, and appends the new assays to
		`self.assays` in sorted order. Because the replacement model is newly
		compiled, the optimizer state from earlier calls to `fit` is not kept.

		Pass in a dictionary of training data and an optional dictionary of
		validation data. The keys to this dictionary are a tuple of the format
		(celltype, assay) and the values are the corresponding track in the
		form of a numpy array. The tracks can either be in the form of an array
		that is in memory or as a memory map. The assays provided should not
		appear in the model.assays attribute, but the cell types should appear
		in the model.celltypes attribute.

		`self.experiments` is not updated with the new experiments: the
		combined list is assigned to the `experiments` attribute of the new
		Keras model instead, so `save` writes the experiments from before this
		call.

		Parameters
		----------
		X_train : dict
			A dictionary of training data values, where the keys are a tuple of
			(celltype, assay) and the values are a track.

		X_valid : dict or None, optional
			A dictionary of validation data values that are used to calculate
			validation set MSE during the training process. If None, validation
			set statistics are not calculated during the training process. On
			Python 3, passing a dictionary raises a TypeError at the end of the
			first epoch; see `avocado.io.data_generator`. Default is None.

		n_epochs : int, optional
			The number of epochs to train on before ending training. Default is 200.

		epoch_size : int, optional
			The number of batches per epoch. Default is 120.

		verbose: int, optional
			The verbosity level of training. Must be one of 0, 1, or 2, where 0
			means silent, 1 means progress bar, and 2 means use only one line
			per epoch. Default is 1.

		callbacks : list or None, optional
			A list of keras callback instances to be called during training. 

		**kwargs : optional
			Any other keyword arguments to be passed into the `fit_generator`
			method.

		Returns
		-------
		history : keras.callbacks.History
			The keras history object of the temporary model, which records
			training loss values and metric values in its `history` attribute.
		"""

		if not isinstance(X_train, dict):
			raise ValueError("X_train must be a dictionary where the keys" \
				" are (celltype, assay) tuples and the values are the track" \
				" corresponding to that pair.")

		if X_valid is not None and not isinstance(X_valid, dict):
			raise ValueError("X_valid must be a dictionary where the keys" \
				" are (celltype, assay) tuples and the values are the track" \
				" corresponding to that pair.")	

		for (celltype, assay), track in X_train.items():
			if celltype not in self.celltypes:
				raise ValueError("Celltype {} appears in the training data " \
					"but not in the list of cell types already in the " \
					"model.".format(celltype))

			if assay in self.assays:
				raise ValueError("Assay {} appears in the training data " \
					"and also in the list of assays already in the " \
					"model.".format(assay))

			if len(track) != self.n_genomic_positions:
				raise ValueError("The track corresponding to {} {} is of " \
					"size {} while the model encodes {} genomic " \
					"positions".format(celltype, assay, len(track), 
						self.n_genomic_positions))

		if X_valid is not None:
			for (celltype, assay), track in X_valid.items():
				if celltype not in self.celltypes:
					raise ValueError("Celltype {} appears in the validation " \
						"data but not in the list of cell types already in " \
						"the model.".format(celltype))

				if assay in self.assays:
					raise ValueError("Assay {} appears in the training data " \
						"and also in the list of assays already in the " \
						"model.".format(assay))

				if len(track) != self.n_genomic_positions:
					raise ValueError("The track corresponding to {} {} is of " \
						"size {} while the model encodes {} genomic " \
						"positions".format(celltype, assay, len(track), 
							self.n_genomic_positions))

		new_assays = list(numpy.unique([assay for _, assay in X_train.keys()]))

		model = build_model(n_celltypes=self.n_celltypes,
							n_celltype_factors=self.n_celltype_factors,
							n_assays=len(new_assays),
							n_assay_factors=self.n_assay_factors,
							n_genomic_positions=self.n_genomic_positions,
							n_25bp_factors=self.n_25bp_factors,
							n_250bp_factors=self.n_250bp_factors,
							n_5kbp_factors=self.n_5kbp_factors,
							n_layers=self.n_layers,
							n_nodes=self.n_nodes,
							freeze_celltypes=True,
							freeze_assays=False,
							freeze_genome_25bp=True,
							freeze_genome_250bp=True,
							freeze_genome_5kbp=True,
							freeze_network=True)

		for old_layer, new_layer in zip(self.model.layers, model.layers):
			if 'input' in old_layer.name:
				continue
			if old_layer.name == 'assay_embedding':
				continue

			new_layer.set_weights(old_layer.get_weights())


		X_train_gen = sequential_data_generator(self.celltypes, new_assays, 
			X_train, self.n_genomic_positions, self.batch_size)

		if X_valid is not None:
			X_valid_gen = data_generator(self.celltypes, new_assays, 
				X_valid, self.n_genomic_positions, self.batch_size)

			history = model.fit_generator(X_train_gen, epoch_size, n_epochs, 
				workers=1,  validation_data=X_valid_gen, 
				validation_steps=30, verbose=verbose, callbacks=callbacks, 
				**kwargs)
		else:
			history = model.fit_generator(X_train_gen, epoch_size, n_epochs, 
				workers=1,  verbose=verbose, 
				callbacks=callbacks, **kwargs)

		for layer in self.model.layers:
			if layer.name == 'assay_embedding':
				assay_embedding = layer.get_weights()[0]
				break

		for layer in model.layers:
			if layer.name == 'assay_embedding':
				new_assay_embedding = layer.get_weights()[0]
				break

		assay_embedding = numpy.concatenate([assay_embedding, 
			new_assay_embedding]) 

		self.assays.extend(new_assays)
		self.n_assays = len(self.assays)

		model = build_model(n_celltypes=self.n_celltypes,
							n_celltype_factors=self.n_celltype_factors,
							n_assays=self.n_assays,
							n_assay_factors=self.n_assay_factors,
							n_genomic_positions=self.n_genomic_positions,
							n_25bp_factors=self.n_25bp_factors,
							n_250bp_factors=self.n_250bp_factors,
							n_5kbp_factors=self.n_5kbp_factors,
							n_layers=self.n_layers,
							n_nodes=self.n_nodes,
							freeze_celltypes=self.freeze_celltypes,
							freeze_assays=self.freeze_assays,
							freeze_genome_25bp=self.freeze_genome_25bp,
							freeze_genome_250bp=self.freeze_genome_250bp,
							freeze_genome_5kbp=self.freeze_genome_5kbp,
							freeze_network=self.freeze_network)

		for old_layer, new_layer in zip(self.model.layers, model.layers):
			if 'input' in old_layer.name:
				continue
			if old_layer.name == 'assay_embedding':
				new_layer.set_weights([assay_embedding])
			else:
				new_layer.set_weights(old_layer.get_weights())

		model.experiments = self.experiments + list(X_train.keys())
		self.model = model
		return history

	def predict(self, celltype, assay, start=0, end=None, verbose=0):
		"""Predict a track of epigenomic data.

		This will predict a track of epigenomic data, resulting in one signal
		value per genomic position modeled. Users pass in the cell type and
		the assay that they wish to impute and receive the track of data.

		Parameters
		----------
		celltype : str
			The cell type (aka biosample) to be imputed. Must be one of the
			elements of `self.celltypes`, or `list.index` raises a ValueError.

		assay : str
			The assay to be imputed. Must be one of the elements of
			`self.assays`, or `list.index` raises a ValueError.

		start : int, optional
			The start position to begin the imputation at. By default this is 0,
			corresponding to the start of the track. The value is which 25 bp
			bin to begin prediction at, not the raw genomic coordinate.

		end : int or None, optional
			The end position to stop making imputations at, exclusive. By default
			this is None, meaning to end at `self.n_genomic_positions.`. 

		verbose : int, optional
			The verbosity level of the prediction. Must be 0 or 1.

		Returns
		-------
		track : numpy.ndarray, shape=(end - start,)
			A track of epigenomic signal value predictions for the specified
			cell type and assay for the considered genomic positions, in the
			space the model was trained in. The pre-trained models were trained
			on arcsinh-transformed -log10 p-values, which `numpy.sinh` undoes.
		"""

		if end is not None and end <= start:
			raise ValueError("When given, the end coordinate must be greater" \
				" than the start coordinate.")

		if end is None:
			end = self.n_genomic_positions

		celltype_idx = self.celltypes.index(celltype)
		assay_idx = self.assays.index(assay)

		celltype_idxs = numpy.ones(end-start) * celltype_idx
		assay_idxs = numpy.ones(end-start) * assay_idx

		genomic_25bp_idxs  = numpy.arange(start, end)
		genomic_250bp_idxs = numpy.arange(start, end) // 10
		genomic_5kbp_idxs  = numpy.arange(start, end) // 200

		X = {
			'celltype_input': celltype_idxs, 
			'assay_input': assay_idxs, 
			'genome_25bp_input': genomic_25bp_idxs, 
			'genome_250bp_input': genomic_250bp_idxs,
			'genome_5kbp_input': genomic_5kbp_idxs
		}
		
		track = self.model.predict(X, batch_size=self.batch_size, 
			verbose=verbose)[:,0]
		
		return track

	def get_params(self):
		"""Meant to collect the first weight matrix of every layer.

		This method does not work. It refers to the undefined names `model`
		and `layers`, so every call raises a NameError, and it has no return
		statement. Use `self.model.get_weights()` or the embedding properties
		instead.
		"""

		params = []
		for layer in model.layers:
			params.append(layers.get_weghts()[0])

	def save(self, name="avocado", separators=(',', ' : '), indent=4):
		"""Serialize the model to disk.

		This function produces two files. The first is a json file that has the
		model hyperparameters associated with it. The second is a h5 file that
		contains the architecture of the neural network model, the weights, and
		the optimizer. The json file holds the cell types, assays, experiments,
		factor counts, layer sizes and batch size, but not the `freeze_*`
		settings, which are kept only as the trainable flags of the layers in
		the h5 file.

		Parameters
		----------
		name : str, optional
			The path prefix to use for the json and the h5 file that are
			stored, which are written to `{name}.json` and `{name}.h5`.
			Default is 'avocado'.

		separators : tuple, optional
			The separators to use in the resulting JSON object. Default is
			(',', ' : ').

		indent : int, optional
			The number of spaces to use in the indent of the JSON. Default
			is 4.

		Returns
		-------
		None
		"""

		d = {
			'celltypes': self.celltypes,
			'assays': self.assays,
			'experiments': self.experiments,
			'n_celltype_factors': self.n_celltype_factors,
			'n_assay_factors': self.n_assay_factors,
			'n_genomic_positions': self.n_genomic_positions,
			'n_25bp_factors': self.n_25bp_factors,
			'n_250bp_factors': self.n_250bp_factors,
			'n_5kbp_factors': self.n_5kbp_factors,
			'n_layers': self.n_layers,
			'n_nodes': self.n_nodes,
			'batch_size': self.batch_size
		}

		d = json.dumps(d, separators=separators, indent=indent)

		with open("{}.json".format(name), "w") as outfile:
			outfile.write(d)

		self.model.save("{}.h5".format(name))

	def load_weights(self, name, verbose=0):
		"""Load serialized weights on a layer-by-layer case.

		Load the weights of a pre-saved model on a layer-by-layer case. This
		method will iterate through the layers of the serialized model and
		this model jointly and set the weights in this model to that of the
		serialized model should the weight matrices be of the same size. Should
		they not be of the same size it will not modify the current weight
		matrix. 

		A primary use of this function should be after an initial model has been
		trained on the Pilot regions and now one is fitting a model to each of
		the chromosomes. The size of the genome factors will differ but the other
		components will remain the same. Correspondingly, the identically sized
		weight matrices are those that should be held constant while the differing
		size weight matrices should differ.

		Parameters
		----------
		name : str
			The path prefix of the saved model. The weights are read from
			`{name}.h5`, which is loaded in full as a Keras model.

		verbose : int, optional
			The verbosity level when loading weights. 0 means silent, 1 means
			notify when a weight matrix has been set, 2 means notify what
			action has been taken on each layer. Default is 0.

		Returns
		-------
		None
		"""

		model = keras.models.load_model("{}.h5".format(name))

		for i, (self_layer, layer) in enumerate(zip(self.model.layers, model.layers)):
			w = layer.get_weights()
			w0 = self_layer.get_weights()
			name = self_layer.name

			if len(w) == 0:
				if verbose == 2:
					print("{} has no weights to set".format(name))
				
				continue

			if w[0].shape != w0[0].shape:
				if verbose == 2:
					print("{} is of different size and not set".format(name))

				continue

			self_layer.set_weights(w)
			if verbose > 0:
				print("{} has been set from serialized model".format(name))

	@classmethod
	def load(self, name, freeze_celltypes=False, freeze_assays=False,
		freeze_genome_25bp=False, freeze_genome_250bp=False, 
		freeze_genome_5kbp=False, freeze_network=False):
		"""Load a model that has been serialized to disk.

		The keras model that is saved to disk does not contain any of the
		wrapper information, such as the names of the cell types and assays,
		so it is read from the json file. This method builds an Avocado
		object from the json file and then replaces its Keras model with the
		one read from the h5 file. If the json file has no experiments list,
		as in the pre-trained Roadmap and ENCODE-Core models on Zenodo,
		`experiments` is empty.

		The `freeze_*` arguments do not freeze any layer of the returned
		model. They are stored on the Avocado object, but the Keras model
		that the constructor builds with them is replaced by the one read from
		the h5 file, whose layers keep the trainable flags they were saved
		with. The stored values are used only when `fit_celltypes` or
		`fit_assays` builds a replacement model.

		Parameters
		----------
		name : str
			The name of the file to load. There must be both a .json and a
			.h5 file with this suffix. For example, if "Avocado" is passed in,
			there must be both a "Avocado.json" and a "Avocado.h5" file to
			be loaded in.

		freeze_celltypes : bool, optional
			Stored on the returned object but not applied to its layers; see
			above. Default is False.

		freeze_assays : bool, optional
			Stored on the returned object but not applied to its layers; see
			above. Default is False.

		freeze_genome_25bp : bool, optional
			Stored on the returned object but not applied to its layers; see
			above. Default is False.

		freeze_genome_250bp : bool, optional
			Stored on the returned object but not applied to its layers; see
			above. Default is False.

		freeze_genome_5kbp : bool, optional
			Stored on the returned object but not applied to its layers; see
			above. Default is False.

		freeze_network : bool, optional
			Stored on the returned object but not applied to its layers; see
			above. Default is False.

		Returns
		-------
		model : Avocado
			An Avocado model.
		"""

		with open("{}.json".format(name), "r") as infile:
			d = json.load(infile)

		if 'experiments' in d:
			experiments = d['experiments']
			del d['experiments']
		else:
			experiments = []

		model = Avocado(freeze_celltypes=freeze_celltypes,
						freeze_assays=freeze_assays,
						freeze_genome_25bp=freeze_genome_25bp,
						freeze_genome_250bp=freeze_genome_250bp,
						freeze_genome_5kbp=freeze_genome_5kbp,
						freeze_network=freeze_network,
						**d)

		model.experiments = experiments
		model.model = keras.models.load_model("{}.h5".format(name))
		return model
