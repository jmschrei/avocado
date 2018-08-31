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

### What is Avocado?

There have recently been many large-scale efforts to better understand human genomics and epigenomics by assaying as many different biological phenomena in as many different human cell types as possible. Three of these efforts are the [Roadmap Epigenomics Consortium](http://www.roadmapepigenomics.org/), the [Encyclopedia of DNA Elements (ENCODE) Project](https://www.encodeproject.org/), and the [International Human Epigenome Consortium](https://ihec-epigenomes.org/). These consortia collect thousands of tracks of genome-wide measurements that include histone modification and transcription factor binding through chromatin immunoprecipitation followed by sequencing (ChIP-seq) experiments, gene expression through RNA-seq, CAGE, and RAMPAGE experiments, nucleotide methylation through whole genome bisulfide sequencing (WGBS), replication timing through Repli-seq experiments, and others. These measurements are frequently organized as a tensor with three orthogonal axes; the cell types / tissues, the epigenomic assays, and the length of the genome. Unfortunately, these measurements are typically noisy and redundant despite the tensor being extremely sparse with far fewer than 1% of potential experiments having been performed.

Avocado is a multi-scale deep tensor factorization model that factorizes this tensor of epigenomic data such that it learns latent representations of the modeled cell types, epigenomic assays, and genomic positions. It is multi-scale because it represents the genome axis using three resolutions; 25 bp, 250 bp, and 5 kbp. It is deep because it replaces the generalized dot product used in a factorization approach with a deep neural network and jointly trains the latent factors and the network weights. Avocado is trained on the task of imputing epigenomic experiments and so the latent factors learn representations of each axis that embed important genomic phenomena while the neural network learns weights that can combine these latent factors in such a manner as to predict the signal value of an epigenomic assay in a specific cell type at genomic position.

<p align="center">
	<img src="figures/Avocado-Schematic.gif" width="650"/>
</p>

### What can Avocado do?


