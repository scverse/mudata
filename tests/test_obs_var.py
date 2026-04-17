from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mudata as md


@pytest.mark.parametrize("mdata", (0, 1), indirect=True)
@pytest.mark.parametrize("pull_on_update", (False, True))
def test_obs_global_columns(mdata: md.MuData, pull_on_update: bool, filepath_h5mu: str | Path):
    mdata.obs.drop(columns=mdata.obs.columns, inplace=True)
    for m, mod in mdata.mod.items():
        mod.obs.drop(columns=mod.obs.columns, inplace=True)
        mod.obs["demo"] = m
    mdata.obs["demo"] = "global"
    if pull_on_update:
        with md.set_options(pull_on_update=pull_on_update):
            del mdata._obshash
            mdata.update()
    if mdata.axis == 0 and pull_on_update:
        assert mdata.obs.columns.to_list() == ["demo"] + [f"{m}:demo" for m in mdata.mod.keys()]
    else:
        assert mdata.obs.columns.to_list() == ["demo"]
    mdata.write(filepath_h5mu)
    mdata_ = md.read(filepath_h5mu)
    assert (mdata_.obs.columns == mdata.obs.columns.values).all()


@pytest.mark.parametrize("mdata", (0, 1), indirect=True)
def test_set_obs_names(mdata: md.MuData):  # https://github.com/scverse/mudata/issues/112
    names = {m: mod.obs_names for m, mod in mdata.mod.items()}
    mdata.obs_names = mdata.obs_names
    for m, mod in mdata.mod.items():
        assert np.all(mod.obs_names == names[m])

    with pytest.raises(ValueError, match="length of provided obs_names"):
        mdata.obs_names = ["a", "b", "c"]

    with pytest.raises(ValueError, match="length of provided annotation"):
        mdata.obs = pd.DataFrame()


def test_obs_vector(mdata: md.MuData):
    assert (mdata.obs["arange"] == mdata.obs_vector("arange")).all()
    with pytest.raises(KeyError, match="There is no key foo in MuData"):
        mdata.obs_vector("foo")


def test_obsmap_writeable(mdata: md.MuData):
    mdata.obsmap["test_writeable"] = np.arange(mdata.n_obs)


@pytest.mark.parametrize("mdata", (0, 1), indirect=True)
@pytest.mark.parametrize("pull_on_update", (False, True))
def test_var_global_columns(mdata: md.MuData, pull_on_update, filepath_h5mu: str | Path):
    mdata.var.drop(columns=mdata.var.columns, inplace=True)
    for m, mod in mdata.mod.items():
        mod.var.drop(columns=mod.var.columns, inplace=True)
        mod.var["demo"] = m
    mdata.var["global"] = "global_var"
    if pull_on_update:
        with md.set_options(pull_on_update=pull_on_update):
            del mdata._varhash
            mdata.update()
    if not pull_on_update:
        assert mdata.var.columns.to_list() == ["global"]
    elif mdata.axis == 0:
        assert mdata.var.columns.to_list() == ["global", "demo"]
    else:
        assert mdata.var.columns.to_list() == ["global"] + [f"{m}:demo" for m in mdata.mod.keys()]
    del mdata.var["global"]
    with md.set_options(pull_on_update=pull_on_update):
        mdata.update()
    if not pull_on_update:
        assert mdata.var.shape[1] == 0
    elif mdata.axis == 0:
        assert mdata.var.columns.to_list() == ["demo"]
    else:
        assert mdata.var.columns.to_list() == [f"{m}:demo" for m in mdata.mod.keys()]
    mdata.write(filepath_h5mu)
    mdata_ = md.read(filepath_h5mu)
    assert list(mdata_.var.columns.values) == list(mdata.var.columns.values)


@pytest.mark.parametrize("mdata", (0, 1), indirect=True)
def test_set_var_names(mdata: md.MuData):  # https://github.com/scverse/mudata/issues/112
    names = {m: mod.var_names for m, mod in mdata.mod.items()}
    mdata.var_names = mdata.var_names
    for m, mod in mdata.mod.items():
        assert np.all(mod.var_names == names[m])

    with pytest.raises(ValueError, match="length of provided var_names"):
        mdata.var_names = ["a", "b", "c"]
    with pytest.raises(ValueError, match="length of provided annotation"):
        mdata.var = pd.DataFrame()


def test_var_vector(rng: np.random.Generator, mdata: md.MuData):
    mdata.var["test"] = rng.uniform(size=mdata.n_vars)
    assert (mdata.var["test"] == mdata.var_vector("test")).all()
    with pytest.raises(KeyError, match="There is no key foo in MuData"):
        mdata.var_vector("foo")
    with pytest.raises(KeyError, match="There is no key assert-boolean-1 in MuData .var but there is one in"):
        mdata.var_vector("assert-boolean-1")


def test_varmap_writeable(mdata: md.MuData):
    mdata.varmap["test_writeable"] = np.arange(mdata.n_vars)


@pytest.mark.parametrize("mdata", (0, 1), indirect=True)
def test_names_make_unique(mdata: md.MuData):
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
