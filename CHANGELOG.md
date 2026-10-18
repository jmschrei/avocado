# Changelog

All notable changes to `avocado-epigenome` are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The project kept no changelog before 0.4.1. The entries for earlier versions were reconstructed by matching each package on PyPI to the commit it was built from and reading the code changes between them; README, notebook and figure changes are not listed.

## [0.4.1] - 2026-10-18

The final release of this code base. Avocado is going to be re-implemented as a simpler package that is the same model conceptually but will not necessarily reproduce this version's outputs exactly, and this version is deprecated. It changes no code: it documents the package as it stands, pins the dependencies to versions that work together, and lists the known issues in the README rather than fixing them.

### Packaging

- `pip install avocado-epigenome` installs the code from 0.3.6 again. Version 0.4.0 was uploaded to PyPI on 2020-09-04 from commit 20bba0a, with the version number changed to 0.4.0 in the uploaded package only. That code still passed `pickle_safe=True` to Keras' `fit_generator`, which Keras 2.4.1 rejects (#13). Version 0.3.6 removed the argument seven months later (#14), but its version number was lower than 0.4.0, so pip kept installing 0.4.0. Version 0.4.1 is the 0.3.6 code plus the documentation and packaging changes below.
- The dependencies have upper bounds. They previously had lower bounds only, so pip installed the newest Keras, which since 2.4 requires TensorFlow and has no Theano backend, and the newest numpy, which Theano 1.0.5 cannot use. On Python 3.8, `pip install avocado-epigenome` resolved Keras 2.15.0 and numpy 1.24.4, and `import avocado` failed with `ModuleNotFoundError: No module named 'tensorflow'` even with `KERAS_BACKEND=theano`. Each bound is a failure observed when it was lifted:
  - `keras >= 2.0.8, < 2.4`: Keras 2.4.3 raises `ImportError: Keras requires TensorFlow 2.2 or higher` on import.
  - `numpy >= 1.14.2, < 1.22`: with numpy 1.22.4, Theano 1.0.5 raises `AttributeError: module 'numpy.distutils.__config__' has no attribute 'blas_opt_info'` on import.
  - `h5py < 3`, now a direct dependency: with h5py 3.11.0, Keras 2.3.1 raises `AttributeError: 'str' object has no attribute 'decode'` when loading a saved model, which breaks `Avocado.load` and every pre-trained model.
  - `python_requires='<3.9'`: h5py 2.10.0, the last release below 3, has no wheels for Python 3.9 or later, and building it from source failed.

  The package was tested with Python 3.8.20, Keras 2.3.1, Theano 1.0.5, h5py 2.10.0, numpy 1.21.6, pandas 2.0.3 and pyBigWig 0.3.22 on the CPU. The tests covered every public method of `Avocado`, the README's training example with fewer epochs, loading and predicting with the chr21 Roadmap and ENCODE Core models from Zenodo, and `avocado-impute` end to end on a small custom model and on the chr21 ENCODE Core model.
- The PyPI page shows the README, which carries the deprecation notice. Earlier releases uploaded no long description.

### Documentation

- The README has a deprecation notice and an installation section with the tested environment. It also says to set `KERAS_BACKEND=theano`: Keras 2.3.1 defaults to the TensorFlow backend, which is not installed, so without it `import avocado` and `avocado-impute` fail with `ModuleNotFoundError: No module named 'tensorflow'`.
- The README lists where each pre-trained model can still be downloaded. noble.gs.washington.edu, from which `avocado-impute` downloads models, returns 403 or 404 for every model except Roadmap, including the default ENCODE Core model, so the README's CLI example downloaded nothing and then failed with `Model file for chromosome chr1 not found`. The example now downloads the ENCODE Core model from Zenodo first (#20). Thanks @Mistbornk!
- The README lists the known issues found while testing this release, including the ones in `avocado.utils` reported in #3. Thanks @keenangraham!
- The README's training example used `celltypes` and `assays` before defining them, and its add-a-cell-type example passed two values to a filename format string with one placeholder, so the filename held the cell type where the assay belonged (#15). Thanks @sry002!
- `cli/README.md` described an older version of `avocado-impute`. Its usage block lacked `-k` and three of the `-m` choices, it said the default model was Roadmap (the default has been ENCODE Core since 0.3.3), and it suggested `-m encode-hg38`, which has not been a valid choice since 0.3.3. It now matches `avocado-impute --help` and also documents the model directory defaults, when chrX is imputed, and that `-n` is ignored with `-f`.
- `build_model`, the three generators in `avocado.io`, `download_bigWig` and `get_params` have docstrings. The docstrings of `fit`, `fit_celltypes` and `fit_assays` gave the defaults of `n_epochs` and `epoch_size` swapped (120 and 200, where the code uses 200 and 120). Those of `fit_celltypes` and `fit_assays` said they return a new model, when they replace `self.model` in place and return the Keras `History` of a temporary model. The docstring of `load` was unfinished. Each known bug is now described in the docstring of the function it affects.

## [0.3.6] - 2021-04-13

### Fixed

- `fit`, `fit_celltypes` and `fit_assays` no longer pass `pickle_safe=True` to Keras' `fit_generator`. Keras 2.4.1 rejects that argument, so every call to them failed (#13, #14). Thanks @akmorrow13!

This version number is lower than that of 0.4.0, which was uploaded earlier, so pip installed 0.4.0 instead of this release until 0.4.1.

## [0.4.0] - 2020-09-04

Built from commit 20bba0a, with the version number changed to 0.4.0 in the uploaded package only. The repository never had a commit with this version number.

### avocado-impute

- Writes bigWig files with pyBigWig, which is now a dependency, instead of writing bedgraph files and converting them with `bedGraphToBigWig` (#10).
- `-m` accepts `encode2018full-hg38`, `entex2019-hg38` and `encode2019mouse-mm10`, and each named model is downloaded from its own directory on noble.gs.washington.edu.
- Adds `-k/--skip_missing` to skip chromosomes whose model files are missing.
- chrX is imputed for every named model except `roadmap-hg19`.

## [0.3.5] - 2019-10-28

### avocado-impute

- Drops the last bin of each chromosome's predictions before writing them, to account for an extra bin that some models have at the end of a chromosome.

## [0.3.4] - 2019-10-28

### avocado-impute

- Deletes each chromosome's model and runs the garbage collector before loading the next one.

## [0.3.3] - 2019-10-27

### avocado-impute

- The default model is `encode2018core-hg38` instead of `roadmap-hg19`. The choice `encode-hg38` is renamed `encode2018core-hg38`, and `custom` is added.

## [0.3.2] - 2019-10-27

### Added

- The `avocado-impute` command line tool, installed as a script.
- `fit` takes `sampling`, which is `'sequential'`, `'permuted'` or `'random'`, and `avocado.io` has `permuted_data_generator`.
- `Avocado.experiments` records the (cell type, assay) pairs a model was trained on; `save` writes it and `load` reads it.

### Fixed

- `avocado.io` no longer imports numba, which was not a dependency.
- `sequential_data_generator` converts `data.values()` to a list, which Python 3 requires before indexing it.

## [0.3.1] - 2019-04-29

### Added

- `predict` takes `start` and `end` to impute a range of bins.
- `fit` takes `input_generator` to replace the default training generator.

### Fixed

- `fit_assays` froze the assay embedding and trained the cell type embedding. It now trains the assay embedding and freezes the cell type embedding.

The GitHub release v0.1.0 ("ISMB 2019 Release", 2019-01-09) tags commit 8e213bb, which lies between 0.3.0 and 0.3.1 and was not uploaded to PyPI.

## [0.3.0] - 2018-10-05

### Added

- `fit_celltypes` and `fit_assays`, which add new cell types or assays to a trained model by training only their embeddings.
- The `celltype_embedding`, `assay_embedding` and `genome_embedding` properties.
- `bedgraph_to_dense` shows a progress bar.

## [0.2.1] - 2018-09-26

### Added

- `freeze_*` arguments to `Avocado` and `build_model`, which stop training of the embeddings or of the neural network.
- `Avocado.load` and `Avocado.load_weights`.
- `fit` checks its inputs and raises a `ValueError` for an unknown cell type or assay or a track of the wrong length.
- `avocado.utils`, with `download_bigWig`, `bedgraph_to_dense` and `decimate_vector`.

### Changed

- The dependencies on numpy, Keras and Theano are lower bounds instead of the exact versions 1.14.2, 2.0.8 and 1.0.1, and pandas and tqdm are added.

## [0.2.0] - 2018-08-31

The first release on PyPI.
