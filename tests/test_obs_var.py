import unittest

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


@pytest.mark.usefixtures("filepath_h5mu")
class TestMuData:
    def test_obs_global_columns(self, mdata, filepath_h5mu):
        for m, mod in mdata.mod.items():
            mod.obs["demo"] = m
        mdata.obs["demo"] = "global"
        mdata.update()
        mdata.pull_obs()
        assert list(mdata.obs.columns.values) == ["demo"] + [f"{m}:demo" for m in mdata.mod.keys()]
        mdata.write(filepath_h5mu)
        mdata_ = mudata.read(filepath_h5mu)
        assert list(mdata_.obs.columns.values) == ["demo"] + [
            f"{m}:demo" for m in mdata_.mod.keys()
        ]

    def test_var_global_columns(self, mdata, filepath_h5mu):
        for m, mod in mdata.mod.items():
            mod.var["demo"] = m
        mdata.update()
        mdata.pull_var()
        mdata.var["global"] = "global_var"
        mdata.update()
        mdata.pull_var()
        assert list(mdata.var.columns.values) == ["demo", "global"]
        del mdata.var["global"]
        mdata.update()
        mdata.pull_var()
        assert list(mdata.var.columns.values) == ["demo"]
        mdata.write(filepath_h5mu)
        mdata_ = mudata.read(filepath_h5mu)
        assert list(mdata_.var.columns.values) == ["demo"]


if __name__ == "__main__":
    unittest.main()
