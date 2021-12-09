import unittest
import pytest

import os

import numpy as np
from scipy.sparse import csr_matrix
from anndata import AnnData
import mudata
from mudata import MuData

@pytest.fixture()
def mdata():
    mod1 = AnnData(np.arange(0, 100, 0.1).reshape(-1, 10))
    mod2 = AnnData(np.arange(101, 2101, 1).reshape(-1, 20))
    mods = {"mod1": mod1, "mod2": mod2}
    # Make var_names different in different modalities
    for m in ["mod1", "mod2"]:
        mods[m].var_names = [f"{m}_var{i}" for i in range(mods[m].n_vars)]
    mdata = MuData(mods)
    yield mdata

@pytest.mark.usefixtures("filepath_h5mu", "filepath2_h5mu")
class TestMuData:
    def test_copy(self, mdata):
        mdata_copy = mdata.copy()
        assert mdata.shape == mdata_copy.shape
        assert np.array_equal(mdata.obs_names.values, mdata_copy.obs_names.values)
        assert np.array_equal(mdata.var_names.values, mdata_copy.var_names.values)
        assert np.array_equal(mdata.obs.columns.values, mdata_copy.obs.columns.values)
        assert np.array_equal(mdata.var.columns.values, mdata_copy.var.columns.values)

    def test_view_copy(self, mdata):
        view_n_obs = 5
        mdata_view = mdata[list(range(view_n_obs)),:]
        assert mdata_view.is_view == True
        assert mdata_view.n_obs == view_n_obs
        mdata_copy = mdata_view.copy()
        assert mdata_copy.is_view == False
        assert mdata_copy.n_obs == view_n_obs

    def test_backed_copy(self, mdata, filepath_h5mu, filepath2_h5mu):
        mdata.write(filepath_h5mu)
        mdata_b = mudata.read_h5mu(filepath_h5mu, backed="r")
        assert mdata_b.n_obs == mdata.n_obs
        mdata_b_copy = mdata_b.copy(filepath2_h5mu)
        assert mdata_b_copy.file._filename.name == os.path.basename(filepath2_h5mu)
