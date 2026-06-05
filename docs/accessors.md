# Accessors and paths

```{eval-rst}
.. module:: mudata.acc
```

[](#mudata.acc) provides [accessors](inv:anndata:*:term#accessor) that create [references](inv:anndata:*:term#reference) to axis-aligned 1D and 2D arrays in [MuData](#mudata.MuData) objects.
See the corresponding [AnnData documentation](inv:anndata:*:doc#accessors).

:::{important}
This functionality requires AnnData 0.13 or later.
:::

The central [accessor](inv:anndata:*:term#accessor) is [](#A).
```{eval-rst}
.. autodata:: A
```
See [](#MuAcc) and [AdAcc](#anndata.acc.AdAcc) for examples of how to use it to create [references](inv:anndata:*:term#reference) (i.e. [AdRefs](#anndata.acc.AdRef)).

```{eval-rst}
.. autosummary::
   :toctree: generated

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
