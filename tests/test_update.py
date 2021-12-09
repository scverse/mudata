import unittest
import pytest

import numpy as np
from anndata import AnnData
from mudata import MuData


@pytest.fixture()
def mdata():
    mod1 = AnnData(X=np.random.normal(size=1000).reshape(-1, 10))
    mod2 = AnnData(X=np.random.normal(size=1000).reshape(-1, 10))
    np.random.seed(100)
    batches = np.random.choice(["a", "b", "c"], size=100, replace=True)
    mods = {"mod1": mod1, "mod2": mod2}
    # Make var_names different in different modalities
    for m in ["mod1", "mod2"]:
        mods[m].var_names = [f"{m}_var{i}" for i in range(mods[m].n_vars)]
        mods[m].obs["min_count"] = mods[m].X.min(axis=1)
    mdata = MuData(mods)
    mdata.obs["batch"] = batches
    yield mdata


@pytest.mark.usefixtures("filepath_h5mu")
class TestMuData:
    def test_update_after_filter_obs_adata(self, mdata, filepath_h5mu):
        """
        Check for muon issue #44.
        """
        # Replicate in-place filterin in muon:
        # mu.pp.filter_obs(mdata['mod1'], 'min_count', lambda x: (x < -2))
        mdata.mod["mod1"] = mdata["mod1"][mdata["mod1"].obs["min_count"] < -2].copy()
        mdata.update()
        assert mdata.obs["batch"].isna().sum() == 0


if __name__ == "__main__":
    unittest.main()
