"""Multimodal datasets"""

try:  # See https://github.com/maresb/hatch-vcs-footgun-example
    from setuptools_scm import get_version

    __version__ = get_version(root="../..", relative_to=__file__)
except (ImportError, LookupError):
    try:
        from ._version import __version__
    except ModuleNotFoundError:
        raise RuntimeError(
            "mudata is not correctly installed. Please install it, e.g. with pip."
        )

from ._core import utils
from ._core.config import set_options
from ._core.io import *
from ._core.merge import concat
from ._core.mudata import MuData
from ._core.to_ import to_anndata, to_mudata

__anndataversion__ = "0.1.0"
__mudataversion__ = "0.1.0"
