"""Multimodal datasets"""

from anndata import AnnData
from scverse_misc import ExtensionNamespace
from scverse_misc import make_register_namespace_decorator as _make_register_namespace_decorator

from ._core import utils
from ._core.config import set_options
from ._core.io import (
    read,
    read_anndata,
    read_h5ad,
    read_h5mu,
    read_zarr,
    write,
    write_anndata,
    write_h5ad,
    write_h5mu,
    write_zarr,
)
from ._core.merge import concat
from ._core.mudata import MuData
from ._core.to_ import to_anndata, to_mudata
from ._version import __version__, __version_tuple__

# file format versions
__anndataversion__ = "0.1.0"
__mudataversion__ = "0.1.0"

register_mudata_namespace = _make_register_namespace_decorator(MuData, "mdata", "register_mudata_namespace", "numpy")

__all__ = [
    "__version__",
    "MuData",
    "AnnData",
    "utils",
    "set_options",
    "to_anndata",
    "to_mudata",
    "concat",
    "read",
    "read_h5ad",
    "read_anndata",
    "read_h5mu",
    "read_zarr",
    "write",
    "write_h5ad",
    "write_anndata",
    "write_h5mu",
    "write_zarr",
]
