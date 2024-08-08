import unittest

import numpy as np
import pytest
from anndata import AnnData

import mudata
from mudata import MuData


N, D1, D2 = 100, 20, 30
N1 = 15
N2 = N - N1


@pytest.fixture()
def mdata():
    mdata = MuData(
        {
            "mod1": AnnData(np.arange(0, 200, 0.1).reshape(-1, D1)),
            "mod2": AnnData(np.arange(101, 3101, 1).reshape(-1, D2)),
        }
    )
    mdata["mod1"].var_names = [f"mod1_var{i}" for i in range(1, D1 + 1)]
    mdata["mod2"].var_names = [f"mod2_var{i}" for i in range(1, D2 + 1)]
    mdata.update()
    return mdata


@pytest.mark.usefixtures("filepath_h5mu", "filepath_zarr")
class TestMuData:
    def test_merge(self, mdata, filepath_h5mu):
        mdata1, mdata2 = mdata[:N1, :].copy(), mdata[N1:, :].copy()
        mdata_ = mudata.concat([mdata1, mdata2])
        assert list(mdata_.mod.keys()) == ["mod1", "mod2"]
        for m in mdata_.mod_names:
            assert mdata_.mod[m].shape == mdata.mod[m].shape
        assert np.array_equal(mdata_.mod["mod1"].X, mdata.mod["mod1"].X)
        assert np.array_equal(mdata_.mod["mod2"].X, mdata.mod["mod2"].X)

    def test_merge_and_write(self, mdata, filepath_h5mu):
        mdata1, mdata2 = mdata[:N1, :].copy(), mdata[N1:, :].copy()
        mdata_merged = mudata.concat([mdata1, mdata2])
        mdata_merged.write_h5mu(filepath_h5mu)
        mdata_ = mudata.read_h5mu(filepath_h5mu)
        assert list(mdata_.mod.keys()) == ["mod1", "mod2"]
        for m in mdata_.mod_names:
            assert mdata_.mod[m].shape == mdata.mod[m].shape
        assert np.array_equal(mdata_.mod["mod1"].X, mdata.mod["mod1"].X)
        assert np.array_equal(mdata_.mod["mod2"].X, mdata.mod["mod2"].X)
