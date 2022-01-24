# Designing `MuData`

This document outlines design considerations and some technical details about `MuData` implementation.

## Multimodal data containers

`MuData` provides an implementation of the data type for storing multimodal measurements. It is embodied in the `MuData` class and represents a particular way of thinking about multimodal objects as _containers_. That means the unimodal measurements are stored in fully functional objects that can be operated independently — those are `AnnData` instances in the case of `MuData`. These containers can also store information that only makes sense when all of its insides are considered together, e.g. embeddings or cell annotation generated on all modalities en masse.

Such design is transparent, builds on existing software, which has been widely adopted by some communities (e.g. `scanpy` for single-cell data analysis), as well as data formats (HDF5-based files), and also has potential to grow gaining container-level features without disrupting current code. The latter being particularly important when contrasting it to an alternative approach of extending existing data formats such as `AnnData`. See e.g. [this anndata-related discussion](https://github.com/theislab/anndata/issues/237) where multiple challenges has been raised such as [how to have](https://github.com/theislab/anndata/issues/237#issuecomment-562505701) both modality-specific and cross-modality APIs (e.g. to create and to store respective embeddings).

One of the great side effects of this _container_ approach is that AnnData objects can be directly read from and written to HDF5 files with multimodal data:

```py
import mudata

# Read from inside the .h5mu file
adata = mudata.read("pbmc_10k.h5mu/rna")

# Write insides the .h5mu file
mudata.write("pbmc_10k.h5mu/rna", adata)
```

One can verify in their terminal it's stored in the HDF5 file as expected:

```sh
h5ls pbmc_10k.h5mu/mod/rna
# X		Group
# obs		Group
# var		Group
# ...
```


## Operating on MuData objects

To handle `MuData` objects with multi-omics data, a [multimodal omics analysis framework (`muon`)](https://github.com/PmBio/muon) has been built. More implementation details can be found [in the respective `DESIGN.md` file](https://github.com/scverse/muon/blob/master/DESIGN.md).
