Say hello to MuData
===================

**MuData** is a format for annotated multimodal datasets. MuData is native to Python but provides cross-language functionality via HDF5-based ``.h5mu`` files.

.. _website: https://scverse.org/
.. _governance: https://scverse.org/about/roles/
.. _NumFOCUS: https://numfocus.org/
.. _donation: https://numfocus.org/donate-to-scverse/

MuData is part of the scverse® project (`website`_, `governance`_) and is fiscally sponsored by `NumFOCUS`_.
Please consider making a tax-deductible `donation`_ to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

.. raw:: html

   <p align="center">
       <a href="https://numfocus.org/project/scverse">
           <img src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png" width="200">
       </a>
   </p>


MuData objects as containers
----------------------------

``mudata`` package introduces multimodal data objects (:class:`mudata.MuData` class) allowing Python users to work with increasigly complex datasets efficiently and to build new workflows and computational tools around it.
::
	MuData object with n_obs × n_vars = 10110 × 110101
	 2 modalities
	  atac: 10110 x 100001
	  rna: 10110 x 10100

MuData objects enable multimodal information to be stored & accessed naturally, embrace `AnnData <https://github.com/theislab/anndata>`_ for the individual modalities, and can be serialized to ``.h5mu`` files. :doc:`Learn more about multimodal objects </io/mudata>` as well as :doc:`file formats for storing & sharing them </io/output>`. 

Natural interface
-----------------

MuData objects feature an AnnData-like interface and familiar concepts such as *observations* and *variables* for the two data dimensions. Get familiar with MuData in the :doc:`Quickstart tutorial </notebooks/quickstart_mudata>`.

Handling MuData objects
-----------------------

A flagship framework for multimodal omics analysis — ``muon`` — has been built around the MuData format. Find more information on it `in its documentation <https://muon.readthedocs.io/en/latest/>`_ and `on the tutorials page <https://muon-tutorials.readthedocs.io/en/latest/>`_ as well as `in the corresponding publication <https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02577-8>`_.


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting started

   notebooks/quickstart_mudata.ipynb
   notebooks/nuances.ipynb
   notebooks/axes.ipynb
   notebooks/annotations_management.ipynb

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Documentation

   install
   io/input
   io/mudata
   io/output
   io/spec
   api/index
   changelog

