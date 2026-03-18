import numpy as np
import pandas as pd
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

    with pytest.raises(ValueError, match="length of provided obs_names"):
        mdata.obs_names = ["a", "b", "c"]

    with pytest.raises(ValueError, match="length of provided annotation"):
        mdata.obs = pd.DataFrame()


def test_obs_vector(mdata):
    assert (mdata.obs["arange"] == mdata.obs_vector("arange")).all()
    with pytest.raises(KeyError, match="There is no key foo in MuData"):
        mdata.obs_vector("foo")


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

    with pytest.raises(ValueError, match="length of provided var_names"):
        mdata.var_names = ["a", "b", "c"]
    with pytest.raises(ValueError, match="length of provided annotation"):
        mdata.var = pd.DataFrame()


def test_var_vector(rng, mdata):
    mdata.var["test"] = rng.uniform(size=mdata.n_vars)
    assert (mdata.var["test"] == mdata.var_vector("test")).all()
    with pytest.raises(KeyError, match="There is no key foo in MuData"):
        mdata.var_vector("foo")
    with pytest.raises(KeyError, match="There is no key assert-boolean-1 in MuData .var but there is one in"):
        mdata.var_vector("assert-boolean-1")


@pytest.mark.parametrize("mdata", (0, 1), indirect=True)
def test_names_make_unique(mdata):
    attr = "obs" if mdata.axis == 0 else "var"
    oattr = "var" if mdata.axis == 0 else "obs"
    namesattr = f"{oattr}_names"
    namesfun = getattr(mdata, f"{oattr}_names_make_unique")

    mods = mdata.mod_names
    names = getattr(mdata.mod[mods[0]], namesattr).to_list()
    names[1] = names[0]
    setattr(mdata.mod[mods[0]], namesattr, names)
    namesfun()
    assert mdata.shape[1 - mdata.axis] == sum(mod.shape[1 - mdata.axis] for mod in mdata.mod.values())
    assert getattr(mdata, namesattr).is_unique

    for mod in mods[:2]:
        names = getattr(mdata.mod[mod], namesattr).to_list()
        names[1] = names[0] = "testname"
        setattr(mdata.mod[mod], namesattr, names)
    with pytest.warns(UserWarning, match="Modality names will be prepended"):
        namesfun()
    assert mdata.shape[1 - mdata.axis] == sum(mod.shape[1 - mdata.axis] for mod in mdata.mod.values())
    assert getattr(mdata, namesattr).is_unique
    for m, mod in mdata.mod.items():
        assert getattr(mod, namesattr).is_unique
        assert (getattr(mod, namesattr).str[: len(m) + 1] == f"{m}:").all()

    with pytest.raises(TypeError, match="axis="):
        getattr(mdata, f"{attr}_names_make_unique")()
