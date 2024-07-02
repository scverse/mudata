import numpy as np
import pytest
from anndata import AnnData

from mudata import MuData

# Dimensions
N = 100
D1, D2 = 10, 20
D = D1 + D2


@pytest.fixture()
def mdata():
    mod1 = AnnData(np.arange(0, 100, 0.1).reshape(-1, D1))
    mod1.obs_names = [f"obs{i}" for i in range(mod1.n_obs)]
    mod1.var_names = [f"var{i}" for i in range(D1)]

    mod21 = AnnData(np.arange(3101, 5101, 1).reshape(-1, D2))
    mod22 = AnnData(np.arange(3101, 5101, 1).reshape(-1, D2))
    # Same obs_names and var_names
    mod21.obs_names = mod1.obs_names.copy()
    mod22.obs_names = mod1.obs_names.copy()
    mod21.var_names = [f"var{i}" for i in range(D1, D)]
    mod22.var_names = [f"var{i}" for i in range(D1, D)]
    mod2 = MuData({"mod21": mod21, "mod22": mod22}, axis=-1)

    mdata = MuData({"mod1": mod1, "mod2": mod2})
    return mdata


class TestMuData:
    def test_nested_mudata(self, mdata):
        assert mdata.shape == (N, D)
        assert mdata["mod1"].shape == (N, D1)
        assert mdata["mod2"].shape == (N, D2)
        assert mdata.axis == 0
        assert mdata["mod2"].axis == -1

    def test_mod_repr(self, mdata):
        assert (
            mdata.mod.__repr__()
            == f"MuData\n├─ mod1 AnnData ({N} x {D1})\n└─ mod2 MuData [shared obs and var] ({N} × 20)\n   ├─ mod21 AnnData ({N} x {D2})\n   └─ mod22 AnnData ({N} x {D2})"
        )
