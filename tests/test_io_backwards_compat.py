from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData
from anndata.tests.helpers import assert_equal

from mudata import MuData, read_h5mu

ARCHIVE_PTH = Path(__file__).parent / "data/archives"


@pytest.fixture(params=list(ARCHIVE_PTH.glob("v*")), ids=lambda x: x.name)
def archive_dir(request):
    return request.param


@pytest.fixture
def mdata():
    return MuData(
        {
            "mod1": AnnData(np.arange(0, 100, 0.1).reshape(-1, 10)),
            "mod2": AnnData(np.arange(101, 2101, 1).reshape(-1, 20)),
        }
    )


class TestLegacyMuData:
    def test_legacy_files(self, archive_dir, mdata):
        from_h5mu = read_h5mu(archive_dir / "mudata.h5mu")

        for mod in mdata.mod:
            assert_equal(from_h5mu.mod[mod].X, mdata.mod[mod].X, exact=False)
