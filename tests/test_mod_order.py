import h5py
import numpy as np
import pytest
from anndata import AnnData

import mudata
from mudata import MuData

# Dimensions
N = 100
D1, D2, D3 = 10, 20, 30
D = D1 + D2 + D3

KEYS_ORDERED = ["c-first", "a-second", "b-third"]


@pytest.fixture()
def mdata():
    mod1 = AnnData(np.arange(0, 100, 0.1).reshape(-1, D1))
    mod1.obs_names = [f"obs{i}" for i in range(mod1.n_obs)]
    mod1.var_names = [f"var{i}" for i in range(D1)]

    mod2 = AnnData(np.arange(3101, 5101, 1).reshape(-1, D2))
    mod2.obs_names = [f"obs{i}" for i in range(mod1.n_obs)]
    mod2.var_names = [f"var{i}" for i in range(D2)]

    mod3 = AnnData(np.arange(5101, 8101, 1).reshape(-1, D3))
    mod3.obs_names = [f"obs{i}" for i in range(mod1.n_obs)]
    mod3.var_names = [f"var{i}" for i in range(D3)]

    # order of modalities should be preserved, namely
    # c-first, then a-second, then b-third
    mdata = MuData({"c-first": mod1, "a-second": mod2, "b-third": mod3})
    return mdata


@pytest.mark.usefixtures("filepath_h5mu")
class TestMuData:
    def test_initial_order(self, mdata):
        mods = list(mdata.mod.keys())
        assert len(mods) == 3
        assert mods == KEYS_ORDERED

    def test_order_on_read(self, mdata, filepath_h5mu):
        mdata.write_h5mu(filepath_h5mu)
        mdata_read = mudata.read_h5mu(filepath_h5mu)

        mods = list(mdata_read.mod.keys())
        assert len(mods) == 3
        assert mods == KEYS_ORDERED

        # Test implementation (storage) as well
        with h5py.File(filepath_h5mu, "r") as f:
            assert "mod-order" in f["mod"].attrs
            assert list(f["mod"].attrs["mod-order"]) == KEYS_ORDERED
