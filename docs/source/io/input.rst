.. MuData documentation master file, created by
   sphinx-quickstart on Thu Oct 22 02:24:42 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Input data
==========

A default way to import ``MuData`` is the following:
::
	from mudata import MuData


There are various ways in which the data can be provided to create a MuData object:


.. contents:: :local:
    :depth: 3

.. toctree::
   :maxdepth: 10

   *


AnnData objects
---------------

MuData object can be constructed from a dictionary of existing AnnData objects:
::
	mdata = MuData({'rna': adata_rna, 'atac': adata_atac})


AnnData objects themselves can be easily constructed from NumPy arrays and/or Pandas DataFrames annotating features (*variables*) and samples/cells (*observations*). This makes it a rather general data format to work with any type of high-dimensional data.
::
	from anndata import AnnData
	adata = AnnData(X=matrix, obs=metadata_df, var=features_df)


Please see more details on how to operate on AnnData objects `in the anndata documentation <https://anndata.readthedocs.io/>`_.


Omics data
----------

When data fromats specific to genomics are of interest, specialised readers can be found in analysis frameworks such as `muon <https://muon.readthedocs.io/>`_. These functions, including the ones for Cell Ranger count matrices as well as Snap files, `are described here <https://muon.readthedocs.io/en/latest/io/input.html>`_.


