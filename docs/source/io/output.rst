Output data
===========

In order to save & share multimodal data, ``.h5mu`` file format has been designed.

.. contents:: :local:
    :depth: 3

.. toctree::
   :maxdepth: 10

   *


.h5mu files
-----------

``.h5mu`` files are the default storage for MuData objects. These are HDF5 files with a standardised structure, which is similar to the one of ``.h5ad`` files where AnnData objects are stored. The most noticeable distinction is ``.mod`` group presence where individual modalities are stored — in the same way as they would be stored in the ``.h5ad`` files.
::
	mdata.write("mudata.h5mu")

Inspect the contents of the file in the terminal:

.. code-block:: console

	> h5ls mudata.h5mu
	mod                      Group
	obs                      Group
	obsm                     Group
	var                      Group
	varm                     Group

	> h5ls data/mudata.h5mu/mod
	atac                     Group
	rna                      Group



AnnData inside .h5mu
^^^^^^^^^^^^^^^^^^^^

Individual modalities in the ``.h5mu`` file are stored in exactly the same way as AnnData objects. This, together with the hierarchical nature of HDF5 files, makes it possible to read individual modalities from ``.h5mu`` files as well as to save individual modalities to the ``.h5mu`` file:
::
	adata = mudata.read("mudata.h5mu/rna")

	mudata.write("mudata.h5mu/rna", adata)

The function :func:`mudata.read` automatically decides based on the input if :func:`mudata.read_h5mu` or rather :func:`mudata.read_h5ad` should be called.

Learn more about the on-disk format specification shared by MuData and AnnData `in the AnnData documentation <https://anndata.readthedocs.io/en/latest/fileformat-prose.html>`_.


.zarr files
-----------

`Zarr <https://zarr.readthedocs.io/en/stable/>`_ is a cloud-friendly format for chunked N-dimensional arrays. Zarr is another supported serialisation format for MuData objects:
::
        mdata = mudata.read_zarr("mudata.zarr")

        mdata.write_zarr("mudata.zarr")

Just as with ``.h5mu`` files, MuData objects saved in ``.zarr`` format resemble how AnnData objects are stored, with one additional level of abstraction:

.. code-block:: console

        > tree -L 1 mudata.zarr
        mudata.zarr
        ├── mod
        ├── obs
        ├── obsm
        ├── obsmap
        ├── obsp
        ├── uns
        ├── var
        ├── varm
        ├── varmap
        └── varp

