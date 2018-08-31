# avocado

Avocado is a multi-scale deep tensor factorization model that is used to learn a latent representation of the human epigenome. The purpose of this model is two fold; first, to impute epigenomic experiments that have not yet been performed, and second, to use the learned latent representation in downstream genomics tasks. The primary project page can be found at https://noble.gs.washington.edu/proj/avocado/ 

This repository is comprised of multiple parts:

(1) A Python package, named Avocado, that makes it easy to train such models and use them. The default parameters are those used in the manuscript.

(2) Unmodified research scripts in the folder "scripts/" that show most of the important scripts for the manuscript. These scripts will have hard-coded file paths and calls to a SGE cluster that will likely not work on another machine, but are provided for transparency and review. The full set of all scripts used during the course of the project can be found at https://noble.gs.washington.edu/~jmschr/avocado/ with data processing scripts found in "scripts/" and exploratory code / results found in various folders under "exps/". 

(3) A Jupyter notebook named "Avocado Training Demo.ipynb" that contains a demonstration of how to train a scaled down version of the model using a CPU on a subset of data, use a trained model to impute experiments not in the training set, and how to extract the latent representation. The data is a subset of the Roadmap compendium, consisting of 25 experiments from 5 cell types and 5 assays, restricted to the ENCODE Pilot Regions. The data can be found as arrays in the file format "data/{celltype}.{assay}.pilot.arcsinh.npz". A .html version of the notebook is also provided as "Avocado Training Demo.html" that is read-only.

(4) A Jupyter notebook named "Avocado Downstream Task Demo.ipynb" that contains a demonstration on how to evaluate various feature sets at downstream genomics tasks. This example contains pre-extracted data from the Roadmap compendium, ChromImpute/PREDICTD/Avocado imputed measurements, and the Avocado latent factors for the prediction of gene expression in IMR90.

1) System Requirements

This code has been tested on a Ubuntu 16.04 machine running 2.7.14 (64-bit). However, it should run on any machine that has Python 2.7 installed.

The following Python packages are required and have been tested:

IPython     5.4.1
pandas      0.21.1
numpy       1.14.2
keras       2.0.8
theano      1.0.1
sklearn     0.19.1
joblib      0.11
tqdm        4.19.4
xgboost     0.7
matplotlib  2.1.2
seaborn     0.8.1

A GPU is not required to run the demo code, but will significantly speed it up. Many GPUs were used to train the full Avocado model.

2) Installation Guide

A call to `python setup.py install` should install the required dependencies. The installation of all packages, should the user have none, should not exceed 10 minutes given a standard internet connection (1 Mb/s).

3) Demo and Instructions for Use

Please see the "Avocado Training Demo.html" and "Avocado Downstream Task Demo.html" for demonstrations with narrative text of how to define and train a model, and how to use the extracted latent factors on a downstream genomics task. Given that training the model presented in the manuscript required multiple GPUs on hundreds of gigabytes of data, we cannot provide instructions for reproducing the quantitative results of the manuscript using this demonstration.
