Release notes
=============

.. contents:: :local:
    :depth: 3

.. toctree::
   :maxdepth: 10

   *

v0.3.0
------

This version comes with a notable change to the way the annotations of individual modalities are treated.
It implements pull/push interface for annotations with functions :func:`mudata.MuData.pull_obs`, :func:`mudata.MuData.pull_var`, :func:`mudata.MuData.push_obs`, and :func:`mudata.MuData.push_var`.

:func:`mudata.MuData.update` performance and behaviour have been generally improved.
For compatibility reasons, this release keeps the old behaviour of pulling annotations on read/update as default.
This will be changed in the next release. In order to adopt the new behaviour, use :func:`mudata.set_options` with `pull_on_update=False`.

This release also comes with new functionalities such as :func:`mudata.to_anndata` and :func:`mudata.to_mudata`. 

:class:`mudata.MuData` objects now have a new ``.mod_names`` attribute. ``MuData.mod`` can be pretty-printed. Readers support ``fsspec``, and :func:`mudata.read_zarr` now supports ``mod-order``. The ``uns`` attribute now properly handled by the views. 

v0.2.4
------

This version brings compatibility with the numpy 2.0.0 release and the future anndata 0.11 release with dtype argument deprecation.

Requires anndata 0.10.8 or newer.

v0.2.3
------

Fixes and improvements for backed objects, views, nested MuData objects, I/O and HTML representation.

Pandas 2.0 compatibility.

v0.2.2
------

Path objects ``pathlib.Path`` now work in :func:`mudata.read`.

v0.2.1
------

This version comes with :func:`mudata.MuData.update` improvements and optimisations.

There is now :func:`mudata.MuData.__len__`. This should make it easier to build MuData into workflows that operate on data containers with length. In practice using :func:`mudata.MuData.n_obs` should be preferred.

In this implementation of MuData, default ``dict`` has replaced ``OrderedDict``, e.g. in the ``.uns`` slot, to improve compatibility with new serialisation versions. As of Python 3.6, dictionaries are insertion-ordered.

v0.2.0
------

This version uses new I/O serialisation of `AnnData v0.8 <https://anndata.readthedocs.io/en/latest/release-notes/index.html#th-march-2022>`_.

Updating a MuData object with :func:`mudata.MuData.update` is even faster in many use cases.

There's `a new axes interface <https://github.com/scverse/mudata/blob/master/docs/source/notebooks/axes.ipynb>`_ that allows to use MuData objects as containers with different shared dimensions.


v0.1.2
------

Updating a MuData object with :func:`mudata.MuData.update` is now much faster.

This version also comes with an improved documentation, including `a new page describing the sharp bits <notebooks/nuances.ipynb>`__.

v0.1.1
------

This version comes with improved stability and bug fixes.

v0.1.0
------

Initial ``mudata`` release with ``MuData`` (:class:`mudata.MuData`), previously a part of the ``muon`` framework.

