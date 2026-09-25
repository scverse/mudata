# Accessors and paths

```{eval-rst}
.. module:: mudata.acc
```

[](#mudata.acc) provides {term}`accessors <anndata:accessor>` that create {term}`references <anndata:reference>` to axis-aligned 1D and 2D arrays in [MuData](#mudata.MuData) objects.
See the corresponding {doc}`AnnData documentation <anndata:accessors>`.

:::{important}
This functionality requires AnnData 0.13.2 or newer.
:::

The central {term}`anndata:accessor` is [](#A).
```{eval-rst}
.. autodata:: A
```
See [](#MuAcc) and [AdAcc](#anndata.acc.AdAcc) for examples of how to use it to create {term}`references <anndata:reference>` (i.e. [AdRefs](#anndata.acc.AdRef)).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: class-accessor

   MuAcc
   MultiModAcc
   ModAcc
   ModMapAcc
   ModMetaAcc
   ModLayerAcc
   ModGraphAcc
   ModMultiAcc
   ModMultiMapAcc
   ModGraphMapAcc
```
