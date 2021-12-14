Nuances
=======

.. contents:: :local:
    :depth: 3

.. toctree::
   :maxdepth: 10

   *

This is *the sharp bits* page for ``mudata``, which provides information on nuances when working with ``MuData`` objects.

Variable names
--------------

``MuData`` is designed with features (variables) being different in different modalities. Hence their names should be unique and different between modalities. In other words, ``.var_names`` are checked for uniqueness across modalities.

This behaviour ensures all the functions are easy to reason about. For instance, if there is a ``var_name`` that is present in both modalities, what happens during plotting a joint embedding coloured by this ``var_name`` is not strictly defined.

Nevertheless, ``MuData`` can accommodate modalities with duplicated ``.var_names``. For the typical workflows, we recommend renaming them manually or calling ``.var_names_make_unique()``.

