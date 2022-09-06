MuData specification [RFC]
==========================

Building on top of the `AnnData spec <https://anndata.readthedocs.io/en/latest/fileformat-prose.html>`_, this document provides details on the ``MuData`` on-disk format. For user-facing features, please see `this document <mudata.rst>`__.
::
        >>> import h5py
        >>> f = h5py.File("citeseq.h5mu")
        >>> list(f.keys())
        ['mod', 'obs', 'obsm', 'obsmap', 'uns', 'var', 'varm', 'varmap']

.. contents:: :local:
    :depth: 3

.. toctree::
   :maxdepth: 10

   *


.mod
----

Modalities are stored in a ``.mod`` group of the ``.h5mu`` file in the alphabetical order. To preserve the order of the modalities, there is an attribute ``"mod-order"`` that lists the modalities in their respective order. If some modalities are missing from that attribute, the attribute is to be ignored.
::
        >>> dict(f["mod-order"])
        {'mod-order': array(['rna', 'protein'], dtype=object)}


.obsmap and .varmap
-------------------

While in practice ``MuData`` relies on ``.obs_names`` and ``.var_names`` to collate global observations and variables, it also allows to disambiguate between items with the same name using integer maps. For example, global observations will have non-zero integer values in ``.obsmap["rna"]`` if they are present in the ``"rna"`` modality. If an observation or a variable is missing from a modality, it will correspond to a ``0`` value.

