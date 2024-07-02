from pathlib import Path

import numpy as np
import pytest
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
    return mdata


@pytest.mark.usefixtures("filepath_h5mu", "filepath2_h5mu")
class TestMuData:
    def test_copy(self, mdata):
        mdata_copy = mdata.copy()
        assert mdata.shape == mdata_copy.shape
        assert np.array_equal(mdata.obs_names.values, mdata_copy.obs_names.values)
        assert np.array_equal(mdata.var_names.values, mdata_copy.var_names.values)
        assert np.array_equal(mdata.obs.columns.values, mdata_copy.obs.columns.values)
        assert np.array_equal(mdata.var.columns.values, mdata_copy.var.columns.values)

    def test_view_attributes(self, mdata):
        mdata_copy = mdata.copy()
        n, d = mdata.n_obs, mdata.n_var
        # Populate attributes
        mdata_copy.uns["uns_key"] = {"key": "value"}
        mdata_copy.obs["obs_column"] = False
        mdata_copy.var["var_column"] = False
        mdata_copy.obsm["obsm_key"] = np.arange(n).reshape(-1, 1)
        mdata_copy.varm["varm_key"] = np.arange(d).reshape(-1, 1)
        mdata_copy.obsp["obsp_key"] = np.arange(n * n).reshape(n, n)
        mdata_copy.varp["varp_key"] = np.arange(d * d).reshape(d, d)

        view_n_obs = 7
        mdata_view = mdata_copy[list(range(view_n_obs)), :]
        assert mdata_view.shape == (view_n_obs, mdata.n_var)
        assert len(mdata_view.mod) == len(mdata_copy.mod)
        # AnnData/MuData interface
        for attr in "obs", "var", "obsm", "varm", "obsp", "varp", "uns":
            assert hasattr(mdata_view, attr)
            assert list(getattr(mdata_view, attr).keys()) == list(getattr(mdata_copy, attr).keys())
        # MuData-specific interface
        for attr in "mod", "axis", "obsmap", "varmap":
            assert hasattr(mdata_view, attr)
        assert mdata_view.axis == mdata_copy.axis

    def test_view_copy(self, mdata):
        view_n_obs = 5
        mdata_view = mdata[list(range(view_n_obs)), :]
        assert mdata_view.is_view
        assert mdata_view.n_obs == view_n_obs
        mdata_copy = mdata_view.copy()
        assert not mdata_copy.is_view
        assert mdata_copy.n_obs == view_n_obs

    def test_view_view(self, mdata):
        view_n_obs = 5
        mdata_view = mdata[list(range(view_n_obs)), :]
        assert mdata_view.is_view
        assert mdata_view.n_obs == view_n_obs

        view_view_n_obs = 2
        mdata_view_view = mdata_view[list(range(view_view_n_obs)), :]
        assert mdata_view_view.is_view
        assert mdata_view_view.n_obs == view_view_n_obs

    def test_backed_copy(self, mdata, filepath_h5mu, filepath2_h5mu):
        mdata.write(filepath_h5mu)
        mdata_b = mudata.read_h5mu(filepath_h5mu, backed="r")
        assert mdata_b.n_obs == mdata.n_obs
        mdata_b_copy = mdata_b.copy(filepath2_h5mu)
        assert mdata_b_copy.file._filename.name == Path(filepath2_h5mu).name
