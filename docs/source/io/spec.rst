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
        >>> dict(f["mod"].attrs)
        {'mod-order': array(['prot', 'rna'], dtype=object)}


.obsmap and .varmap
-------------------

While in practice ``MuData`` relies on ``.obs_names`` and ``.var_names`` to collate global observations and variables, it also allows to disambiguate between items with the same name using integer maps. For example, global observations will have non-zero integer values in ``.obsmap["rna"]`` if they are present in the ``"rna"`` modality. If an observation or a variable is missing from a modality, it will correspond to a ``0`` value.
::
        >>> list(f["obsmap"].keys())
        ['prot', 'rna']
        >>> import numpy as np
        >>> np.array(f["obsmap"]["rna"])
        array([   1,    2,    3, ..., 3889, 3890, 3891], dtype=uint32)
        >>> np.array(f["obsmap"]["prot"])
        array([   1,    2,    3, ..., 3889, 3890, 3891], dtype=uint32)

        >>> list(f["varmap"].keys())
        ['prot', 'rna']
        >>> np.array(f["varmap"]["rna"])
        array([    0,     0,     0, ..., 17804, 17805, 17806], dtype=uint32)
        >>> np.array(f["varmap"]["prot"])
        array([1, 2, 3, ..., 0, 0, 0], dtype=uint32)

.axis
-----

Axis describes which dimensions are shared: observations (``axis=0``), variables (``axis=1```), or both (``axis=-1``). It is recorded in the ``axis`` attribute of the file:
::
        >>> f.attrs["axis"]
        0

Multimodal datasets, which have observations shared between modalities, will have ``axis=0``. If no axis attribute is available such as in files with the older versions of this specification, it is assumed to be ``0`` by default.
