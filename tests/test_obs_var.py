import unittest

import numpy as np
import pytest
from anndata import AnnData

import mudata
from mudata import MuData


@pytest.fixture(params=(0, 1))
def mdata(rng, request):
    axis = request.param
    attr = "obs" if axis == 0 else "var"
    oattr = "var" if axis == 0 else "obs"

    mod1 = AnnData(np.arange(0, 100, 0.1).reshape(-1, 10))
    mod2 = AnnData(np.arange(101, 2101, 1).reshape(-1, 20))
    mods = {"mod1": mod1, "mod2": mod2}
    for modname, mod in mods.items():
        setattr(
            mod,
            f"{attr}_names",
            [f"{attr}_{i}" for i in rng.choice(mod.shape[axis], size=mod.shape[axis], replace=False)],
        )
        setattr(mod, f"{oattr}_names", [f"{modname}_{oattr}_{i}" for i in range(mod.shape[1 - axis])])
    mdata = MuData(mods, axis=axis)
    return mdata


@pytest.mark.usefixtures("filepath_h5mu")
class TestMuData:
    def test_obs_global_columns(self, mdata, filepath_h5mu):
        for m, mod in mdata.mod.items():
            mod.obs["demo"] = m
        mdata.obs["demo"] = "global"
        mdata.update()
        if mdata.axis == 0:
            assert list(mdata.obs.columns.values) == [f"{m}:demo" for m in mdata.mod.keys()] + ["demo"]
        else:
            assert list(mdata.obs.columns.values) == ["demo"]
        mdata.write(filepath_h5mu)
        mdata_ = mudata.read(filepath_h5mu)
        assert list(mdata_.obs.columns.values) == list(mdata.obs.columns.values)

    def test_set_obs_names(self, mdata):  # https://github.com/scverse/mudata/issues/112
        names = {m: mod.obs_names for m, mod in mdata.mod.items()}
        mdata.obs_names = mdata.obs_names
        for m, mod in mdata.mod.items():
            assert np.all(mod.obs_names == names[m])

    def test_var_global_columns(self, mdata, filepath_h5mu):
        for m, mod in mdata.mod.items():
            mod.var["demo"] = m
        mdata.update()
        mdata.var["global"] = "global_var"
        mdata.update()
        if mdata.axis == 0:
            assert list(mdata.var.columns.values) == ["demo", "global"]
        else:
            assert list(mdata.var.columns.values) == [f"{m}:demo" for m in mdata.mod.keys()] + ["global"]
        del mdata.var["global"]
        mdata.update()
        if mdata.axis == 0:
            assert list(mdata.var.columns.values) == ["demo"]
        else:
            assert list(mdata.var.columns.values) == [f"{m}:demo" for m in mdata.mod.keys()]
        mdata.write(filepath_h5mu)
        mdata_ = mudata.read(filepath_h5mu)
        assert list(mdata_.var.columns.values) == list(mdata.var.columns.values)

    def test_set_var_names(self, mdata):  # https://github.com/scverse/mudata/issues/112
        names = {m: mod.var_names for m, mod in mdata.mod.items()}
        mdata.var_names = mdata.var_names
        for m, mod in mdata.mod.items():
            assert np.all(mod.var_names == names[m])


if __name__ == "__main__":
    unittest.main()
