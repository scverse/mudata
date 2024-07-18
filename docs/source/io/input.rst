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


Remote storage
--------------

MuData objects can be read and cached from remote locations including via HTTP(S) or from S3 buckets. This is achieved via [`fsspec`](https://github.com/fsspec/filesystem_spec). For example, to read a MuData object from a remote server:
::
   import fsspec

   fname = "https://github.com/gtca/h5xx-datasets/raw/main/datasets/minipbcite.h5mu?download="
   with fsspec.open(fname) as f:
      mdata = mudata.read_h5mu(f)


A caching layer can be added in the following way:
::
   fname_cached = "filecache::" + fname
   with fsspec.open(fname_cached, filecache={'cache_storage': '/tmp/'}) as f:
      mdata = mudata.read_h5mu(f)


For more `fsspec` usage examples see [its documentation](https://filesystem-spec.readthedocs.io/).

S3
^^

MuData objects in the ``.h5mu`` format stored in an S3 bucket can be read with ``fsspec`` as well:
::
   storage_options = {
      'endpoint_url': 'localhost:9000',
      'key': 'AWS_ACCESS_KEY_ID',
      'secret': 'AWS_SECRET_ACCESS_KEY',
   }

   with fsspec.open('s3://bucket/dataset.h5mu', **storage_options) as f:
      mudata.read_h5mu(f)


MuData objects stored in the ``.zarr`` format in an S3 bucket can be read from a *mapping*:
::
   import s3fs

   s3 = s3fs.S3FileSystem(**storage_options)
   store = s3.get_mapper('s3://bucket/dataset.zarr')
   mdata = mudata.read_zarr(store)
