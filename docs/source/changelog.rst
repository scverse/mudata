Release notes
=============

.. contents:: :local:
    :depth: 3

.. toctree::
   :maxdepth: 10

   *

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

