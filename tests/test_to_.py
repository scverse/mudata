import numpy as np
import pytest
from anndata import AnnData

from mudata import MuData, to_mudata

# Dimensions
N = 100
D1, D2, D3 = 10, 20, 30
D = D1 + D2 + D3


@pytest.fixture()
def mdata():
    mod1 = AnnData(np.arange(0, 100, 0.1).reshape(-1, D1))
    mod1.obs_names = [f"obs{i}" for i in range(mod1.n_obs)]
    mod1.var_names = [f"var{i}" for i in range(D1)]
    mod1.var["feature_type"] = "mod1"

    mod2 = AnnData(np.arange(3101, 5101, 1).reshape(-1, D2))
    mod2.obs_names = [f"obs{i}" for i in range(mod1.n_obs)]
    mod2.var_names = [f"var{i}" for i in range(D2)]
    mod2.var["feature_type"] = "mod2"

    mod3 = AnnData(np.arange(5101, 8101, 1).reshape(-1, D3))
    mod3.obs_names = [f"obs{i}" for i in range(mod1.n_obs)]
    mod3.var_names = [f"var{i}" for i in range(D3)]
    mod3.var["feature_type"] = "mod3"

    mdata = MuData({"mod1": mod1, "mod2": mod2, "mod3": mod3})
    mdata.obs["arange"] = np.arange(mdata.n_obs)
    return mdata


class TestMuData:
    def test_to_anndata_simple(self, mdata):
        adata = mdata.to_anndata()
        assert isinstance(adata, AnnData)
        assert adata.shape == mdata.shape
        assert "arange" in adata.obs.columns
        assert np.array_equal(adata.obs["arange"], np.arange(mdata.n_obs))

    def test_to_mudata_simple(self, mdata):
        adata = mdata.to_anndata()
        mdata_from_adata = to_mudata(adata, axis=0, by="feature_type")
        assert isinstance(mdata_from_adata, MuData)
        assert mdata_from_adata.shape == adata.shape
        assert mdata_from_adata.shape == mdata.shape
