# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.1.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.3.2]

### Fixed

- Fixed an [issue](https://github.com/scverse/mudata/issues/99) in [`update()`](#mudata.MuData.update)

## [0.3.1]

### Fixed

- compatibility with anndata 0.10.9

## [0.3.0]

### Added

- Pull/push interface for annotations: [`pull_obs()`](#mudata.MuData.pull_obs), [`pull_var()`](#mudata.MuData.pull_var), [`push_obs()`](#mudata.MuData.push_obs), [`push_var()`](#mudata.MuData.push_var)
- Conversion functions: [`to_anndata()`](#mudata.MuData.to_anndata), [`to_mudata()`](#mudata.to_mudata)
- [Concatenation](#mudata.concat) of MuData objects
- `MuData.mod_names` attribute
- Pretty-printing for `MuData.mod`
- `fsspec` support for readers.

### Fixed

- Improved performance and behavior of [`update()`](#mudata.MuData.update).
  For compatibility reasons, this release keeps the old behaviour of pulling annotations on read/update as default.
- [`read_zarr()`](#mudata.read_zarr) now supports `mod-order`
- Correct handling of the `uns` attribute by views.

### Note

If you want to adopt the new update behaviour, set `mudata.set_options(pull_on_update=False)`.
This will be the default behaviour in the next release.
With it, the annotations will not be copied from the modalities on [`update()`](#mudata.MuData.update) implicitly.

To copy the annotations explicitly, you will need to use [`pull_obs()`](#mudata.MuData.pull_obs) and/or [`pull_var()`](#mudata.MuData.pull_var).

## [0.2.4]

### Changed

- Requires anndata 0.10.8 or newer.

### Fixed

- Compatibility with numpy 2.0
- Compatibility with anndata 0.11

## [0.2.3]

### Fixed

- Fixes and improvements for backed objects, views, nested MuData objects, I/O and HTML representation.
- Pandas 2.0 compatibility

## [0.2.2]

### Fixed

- [`Path`](#pathlib.Path) objects now work in [](#mudata.read)

## [0.2.1]

### Added

- [`MuData.__len__`](#mudata.MuData.__len__).
  This should make it easier to build MuData into workflows that operate on data containers with length.
  In practice using [`n_obs`](#mudata.MuData.n_obs) should be preferred.

### Changed

- Default `dict` has replaced `OrderedDict`, e.g. in the `uns` slot, to improve compatibility with new serialisation versions.
  As of Python 3.6, dictionaries are insertion-ordered.

### Fixed

- Improvements and optimizations to [`update()`](#mudata.MuData.update)

## [0.2.0]

### Added

- [new axes interface](https://github.com/scverse/mudata/blob/master/docs/source/notebooks/axes.ipynb) that allows to use MuData objects as containers with different shared dimensions.

### Changed

- new I/O serialisation of [AnnData v0.8](https://anndata.readthedocs.io/en/latest/release-notes/index.html#th-march-2022)_.

### Fixed

- Updating a MuData object with `MuData.update()` is even faster in many use cases.

## [0.1.2]

### Changed

- Improved documentation, including [a new page describing the sharp bits](notebooks/nuances.ipynb)

### Fixed

- Updating a MuData object with [`update()`](#mudata.MuData.update) is now much faster.

## [0.1.1]

- Various stability and bug fixes

## [0.1.0]

Initial `mudata` release with [`MuData`](#mudata.MuData), previously a part of the `muon` framework.

[0.3.2]: https://github.com/scverse/mudata/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/scverse/mudata/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/scverse/mudata/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/scverse/mudata/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/scverse/mudata/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/scverse/mudata/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/scverse/mudata/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/scverse/mudata/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/scverse/mudata/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/scverse/mudata/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/scverse/mudata/releases/tag/v0.1.0
