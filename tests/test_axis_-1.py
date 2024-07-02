import unittest

import numpy as np
import pytest
from anndata import AnnData

from mudata import MuData


@pytest.mark.usefixtures("filepath_h5mu")
class TestMuData:
    def test_create(self):
        n, d_raw, d_preproc = 100, 900, 300

        a_raw = AnnData(np.random.normal(size=(n, d_raw)))
        a_preproc = a_raw[
            :, np.sort(np.random.choice(np.arange(d_raw), d_preproc, replace=False))
        ].copy()

        mdata = MuData({"raw": a_raw, "preproc": a_preproc}, axis=-1)

        assert mdata.n_obs == n
        assert mdata.n_vars == d_raw


if __name__ == "__main__":
    unittest.main()
