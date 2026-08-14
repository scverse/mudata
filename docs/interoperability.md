# Interoperability

The on-disk representation of mudata files can be read from other languages. Here we list interfaces for working with MuData from your language of choice:

## R

- [MuData](https://bioconductor.org/packages/release/bioc/html/MuData.html) provides IO for `AnnData` and `MuData` stored in HDF5 to Bioconductor's `SingleCellExperiment` and `MultiAssayExperiment` objects.
- [MuDataSeurat](https://pmbio.github.io/MuDataSeurat/) provides IO from `AnnData` and `MuData` stored in HDF5 to `Seurat` objects.

## Julia

- [Muon.jl](https://scverse.org/Muon.jl) provides Julia implementations of `AnnData` and `MuData` objects, as well as IO for the HDF5 and Zarr formats.
