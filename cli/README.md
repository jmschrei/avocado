### Usage

`avocado-impute` is a command line interface (CLI) that uses a pre-trained Avocado model to make imputations. This program handles the loading of and imputation using models for each chromosome and outputs a single bigwig file for each experiment. These bigwig files contain the genome-wide signal for the specified experiment at 25-bp resolution.

This tool gets automatically installed when you use `pip` to install avocado-epigenome v0.3.2 or later (`pip install avocado-epigenome` should work) and can be used on the command line anywhere. It is the simplest way to make imputations from a pre-trained model. For example:

```
avocado-impute -c J004 -a ChIP-seq_ELF1_signal_p-value
```

will make imputations for elongation factor 1 (ELF1) in K562 without any fuss. The output will be a bigwig file `J004.ChIP-seq_ELF1_signal_p-value.bigwig` in the same directory that the file was run.

A full list of cell types (biosamples) and assays that imputations can be made for can be found in the encode2018core-biosamples and encode2018core-assays, if you're using the encode2018core model (the default) or in Roadmap-biosamples and Roadmap-assays if you're using the Roadmap model.

The high level usage instructions are as follows:

```
usage: avocado-impute [-h] [-f FILENAME] [-c CELLTYPE] [-a ASSAY]
                      [-s CHROM_SIZES]
                      [-m {roadmap-hg19,encode2018core-hg38,custom}]
                      [-p MODEL_PATH] [-r] [-n NAME] [-v] [-x] [-d DEVICE]

Avocado is a deep tensor factorization method for modeling large compendia of
epigenomic and transcriptomic data sets. Avocado organizes the compedia into a
three dimensional tensor with the axes being cell types, assays, and the full
length of the genome. By factorizing the data, Avocado is able to learn low
dimensional "latent" representations of each axis independently of the other
axes. This allows the model to impute any genome-wide epigenomics experiments
that has not yet been performed. This program will allow you to use pre-
trained models to make imputations of genome-wide tracks for any pair of cell
type and assay that the model was trained on. See the GitHub repository for a
list of cell lines and assays included in each model.

optional arguments:
  -h, --help            show this help message and exit
  -f FILENAME, --filename FILENAME
                        The name of a tab-delimited file that has all
                        experiments to impute, one per row, with the cell type
                        first followed by the assay. Optionally, there can be
                        a third column for the name of the file.
  -c CELLTYPE, --celltype CELLTYPE
                        The cell type (biosample) to make the imputation in.
  -a ASSAY, --assay ASSAY
                        The type of assay to impute.
  -s CHROM_SIZES, --chrom_sizes CHROM_SIZES
                        A file of chromosome sizes to be used by
                        bedGraphToBigWig, e.g., hg19.chrom.sizes
  -m {roadmap-hg19,encode2018core-hg38,custom}, --model {roadmap-hg19,encode2018core-hg38,custom}
                        The Avocado model to use, either trained on data from
                        the Roadmap compendium aligned to hg19, or from the
                        ENCODE compendium aligned to hg38. Note that this will
                        set the default value for `chrom-sizes` unless a value
                        is provided.
  -p MODEL_PATH, --model_path MODEL_PATH
                        The path where the model files are stored.
  -r, --arcsinh         Whether the imputations are arcsinh-transformed -log10
                        p-values, which is the space that the model is trained
                        in, or the original -log10 p-value space. Default is
                        -log10 p-values.
  -n NAME, --name NAME  The name of the resulting imputation bigwig. Default
                        is <celltype>.<assay>.bigwig
  -v, --verbose         Whether to print out log messages during the
                        imputation.
  -x, --include_x       Whether to make imputations for chrX. Default is
                        False.
  -d DEVICE, --device DEVICE
                        A flag that is passed to THEANO_FLAGS indicating what
                        device to run the imputation on. Default is 'cuda'.
```

### Detailed Instructions

#### Specifying a model to use
The tool can make imputations using any Avocado pre-trained model, including those provided from our manuscripts and also those trained locally by the user. By default this tool will download and then use the model that has been pre-trained using the Roadmap compendium (which uses data mapped to hg19). However, the user can pass in the flag `-m encode-hg38` to download and use a model that has been pre-trained using the larger ENCODE compendium. Additionally, the user can pass in a local directory where a model lives to use that model instead using the flag `-p`, e.g. `-p models/new-avocado-model/` where the directory `models/new-avocado/model` has Avocado models trained for chr1-22 and optionally chrX. 

#### Making imputations for many experiments at once
Unfortunately, because the model file is divided across chromosomes, it can take time to read each file and move the model weights to a GPU. If these weights have to be loaded separately for each experiment it could slow down the speed at which imputations could be made. More concretely, imputing a single experiment using a HDD and a GTX2080 took ~380 seconds to load and move model weights, ~180 seconds to make the imputations, ~280 seconds to write the imputations to a bedgraph file, and ~190 seconds to convert the bedgraph file to a bigwig file. 

Rather than make imputations one at a time, the tool can impute multiple experiments at once if the user passes in a file of experiments using the `-f` flag. The file should have two tab-delimited columns where the first column contains the cell types and the second column contains the assays to impute. Optionally, a third column can be included that gives the name for each file. This process would involve loading up the model file for a chromosome, imputing multiple experiments, writing each out to their own bedgraph files, and then converting each bedgraph file to a bigwig file at the end. Loading up each model file once, rather than once per experiment, can save a significant amount of time. 

If you're planning on making many imputations it is also advised to use a solid-state drive (SSD). This can significantly reduce the amount of time it takes to load the model files and to write out the bedgraph files.

#### Making imputations without a GPU
By default, this tool will assume that you have a GPU and would like to use it. However, you can also use the tool if you do only have a CPU. This can be done by passing in `-d CPU` or any other valid `THEANO_FLAGS` option. Be warning that it is possible to make imputations using only CPUs but that it will be significantly slower.

#### Should the imputations be arcsinh-transformed?
The Avocado model is trained using arcsinh-transformed data and will make imputations in that space (see the paper for reasoning). However, this transformation can be easily undone with the hyperbolic sin (sinh), returning the imputations to the -log10 p-value space. Because it is more common for epigenomic signals to be -log10 p-values (rather than arcsinh-transformed -log10 p-values) this tool will return imputations in that space. If the user  would like imputations in the arcsinh-transformed space they can simply pass in the `-r` flag.
