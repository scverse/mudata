from importlib.metadata import version

import numpy as np
import pytest
from anndata import AnnData
from packaging.version import Version

from mudata import MuData


@pytest.mark.skipif(
    Version(version("anndata")) < Version("0.14.0.dev0"), reason="anndata version too old, no protocols"
)
def test_protocol_compatibility() -> None:
    from anndata.types import AnnDataLikeMetadataOnly

    # the annotated assignment is checked by mypy, the assert at runtime
    m: AnnDataLikeMetadataOnly = MuData({"a": AnnData(np.zeros((3, 2)))})  # type: ignore[assignment] # fmt: skip # FIXME: mypy bug, see discussion in https://github.com/scverse/anndata/pull/2769, remove when switching to a different type checker
    assert isinstance(m, AnnDataLikeMetadataOnly)
