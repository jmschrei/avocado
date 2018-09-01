# avocado

<p align="center">
	<img src="figures/Avocado-Schematic.png" width="650"/>
</p>

Avocado is a multi-scale deep tensor factorization model that is used to learn a latent representation of the human epigenome. The purpose of this model is two fold; first, to impute epigenomic experiments that have not yet been performed, and second, to use the learned latent representation in downstream genomics tasks. The primary project page with links to the full set of imputations can be found at https://noble.gs.washington.edu/proj/avocado/. The manuscript is currently under review at *Nature Methods* but the preprint can be found [here](https://www.biorxiv.org/content/early/2018/07/08/364976).

### Installation

The package can be installed using pip.

```
pip install avocado-epigenome
```

### What can Avocado do?

Avocado can impute epigenomic experiments that have not yet been performed with higher accuracy than either ChromImpute or PREDICTD, two previous methods. These imputations are at 25 bp resolution and cover the full span of chromosomes 1 through 22. The example below shows the accuracy of Avocado's imputations on one particular track of data as the model trains over 400 epochs.

<p align="center">
	<img src="figures/Avocado-Training.gif" width="750"/>
</p>

Avocado's learned latent representation can be used in the place of epigenomic data as input to machine learning models that are predicting some other genomic phenomena, such as promoter-enhancer interactions or chromatin conformation. In almost all cases machine learning models that use the Avocado latent factors outperform those that use epigenomic data from the cell type of interest.

<p align="center">
	<img src="figures/Avocado-tasks.png" width="750"/>
</p>

### How can I use Avocado?

Using Avocado is easy! We can create the model just by passing in a list of cell types, a list of assays, and specifying the various hyperparameters.

```python
from avocado import Avocado

model = Avocado(celltypes, assays, n_layers=1, n_nodes=128, n_assay_factors=24, 
				n_celltype_factors=32, n_25bp_factors=10, n_250bp_factors=20, 
				n_5kbp_factors=30, batch_size=10000)
```

The format of the training data is that of a dictionary where the keys are (cell type, assay) pairs and the value is the corresponding track of epigenomic data.


```python
celltypes = ['E003', 'E017', 'E065', 'E116', 'E117']
assays = ['H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K4me1']

data = {}
for celltype, assay in itertools.product(celltypes, assays):
    filename = 'data/{}.{}.pilot.arcsinh.npz'.format(celltype, assay)
    data[(celltype, assay)] = numpy.load(filename)['arr_0']
```

Now you can fit your model to that data for some number of epochs, where an epoch is defined as some number of batches.

```python
model.fit(data, n_epochs=10, epoch_size=100)
```

After you're done fitting your model you can then impute any track from the cell types and assays that you trained on. In this case we trained on all tracks, but this can be as dense or sparse as one would like as long as there is at least one example of each cell type and assay.

```python
track = model.predict("E065", "H3K4me3")
```

There are currently two tutorials in the form of Jupyter notebooks. One focuses on how to use this code to train an Avocado model, make imputations, and extract the resulting latent factors. The second shows how one might use the latent factors to make predictions in two downstream tasks. 

### Can I add my own cell type and assay to your pre-trained model?

Yes! Because neural networks are flexible it is easy to freeze the existing parameters of the model and add in additional cell types or assays, training only the latent factors corresponding to that cell type or assay. When those latent factors are trained, the model is now able to impute tracks for all assays of the new cell type, or tracks for all cell types given a new assay.

