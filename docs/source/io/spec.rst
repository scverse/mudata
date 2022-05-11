MuData specification [RFC]
==========================

Building on top of the `AnnData spec <https://anndata.readthedocs.io/en/latest/fileformat-prose.html>`_, this document provides details on the ``MuData`` on-disk format. For user-facing features, please see `are described here <mudata.rst>`__.

.. contents:: :local:
    :depth: 3

.. toctree::
   :maxdepth: 10

   *


.mod
----

Modalities are stored in a ``.mod`` group of the ``.h5mu`` file in the alphabetical order. To preserve the order of the modalities, this is an attribute ``"mod-order"`` that lists the modalities in their respective order. If some modalities are missing from that attribute, the attribute is to be ignored.
::
        >>> import h5py
        >>> f = h5py.File("citeseq.h5mu")
        >>> dict(f["mod-order"])
        {'mod-order': array(['rna', 'protein'], dtype=object)}



