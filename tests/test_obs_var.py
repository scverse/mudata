import numpy as np
import pytest

import mudata


@pytest.mark.parametrize("mdata", (0, 1), indirect=True)
def test_obs_global_columns(mdata, filepath_h5mu):
    mdata.obs.drop(columns=mdata.obs.columns, inplace=True)
    for m, mod in mdata.mod.items():
        mod.obs.drop(columns=mod.obs.columns, inplace=True)
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


@pytest.mark.parametrize("mdata", (0, 1), indirect=True)
def test_set_obs_names(mdata):  # https://github.com/scverse/mudata/issues/112
    names = {m: mod.obs_names for m, mod in mdata.mod.items()}
    mdata.obs_names = mdata.obs_names
    for m, mod in mdata.mod.items():
        assert np.all(mod.obs_names == names[m])


@pytest.mark.parametrize("mdata", (0, 1), indirect=True)
def test_var_global_columns(mdata, filepath_h5mu):
    mdata.var.drop(columns=mdata.var.columns, inplace=True)
    for m, mod in mdata.mod.items():
        mod.var.drop(columns=mod.var.columns, inplace=True)
        mod.var["demo"] = m
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


@pytest.mark.parametrize("mdata", (0, 1), indirect=True)
def test_set_var_names(mdata):  # https://github.com/scverse/mudata/issues/112
    names = {m: mod.var_names for m, mod in mdata.mod.items()}
    mdata.var_names = mdata.var_names
    for m, mod in mdata.mod.items():
        assert np.all(mod.var_names == names[m])
