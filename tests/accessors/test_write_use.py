from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Concatenate, Literal

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from anndata.acc import A as adA
from anndata.tests.helpers import assert_equal
from packaging.version import Version

from mudata import MuData
from mudata._core import io
from mudata.acc import A as mdA

if (anndata_version := Version(metadata.version("anndata"))) < Version("0.14dev0"):
    pytest.skip("anndata version too old, no 'use' support in write functions", allow_module_level=True)


@pytest.fixture(scope="module", params=("hdf5", "zarr"))
def file_format(request: pytest.FixtureRequest) -> Literal["hdf5", "zarr"]:
    return request.param


@pytest.fixture(scope="module")
def write_func(file_format: Literal["hdf5", "zarr"]) -> Callable[Concatenate[str | Path, MuData, ...], None]:
    if file_format == "hdf5":
        return io.write_h5mu
    else:
        return io.write_zarr


@pytest.fixture(scope="module")
def read_func(file_format: Literal["hdf5", "zarr"]) -> Callable[[str | Path], MuData]:
    if file_format == "hdf5":
        return io.read_h5mu
    else:
        return io.read_zarr  # type: ignore[return-value]


@pytest.fixture
def filepath(file_format: Literal["hdf5", "zarr"], request: pytest.FixtureRequest) -> Path:
    if file_format == "hdf5":
        return request.getfixturevalue("filepath_h5mu")
    else:
        return request.getfixturevalue("filepath_zarr")


def test_write_mudata_obs_column(
    mdata: MuData,
    write_func: Callable[Concatenate[str | Path, MuData, ...], None],
    read_func: Callable[[str | Path], MuData],
    filepath: Path,
) -> None:
    write_func(filepath, mdata)

    mdata.obs["arange"] *= 5
    mdata.obs["unwritten"] = 1

    write_func(filepath, mdata, use=adA.obs["arange"])
    mdata_ = read_func(filepath)
    assert (mdata.obs["arange"] == mdata_.obs["arange"]).all()
    assert "unwritten" not in mdata_.obs
    for mod in mdata.mod.values():
        assert "arange" not in mod.obs
        assert "unwritten" not in mod.obs

    write_func(filepath, mdata, use=mdA.obs["unwritten"])
    mdata_ = read_func(filepath)
    assert (mdata.obs["unwritten"] == mdata_.obs["unwritten"]).all()
    for mod in mdata.mod.values():
        assert "arange" not in mod.obs
        assert "unwritten" not in mod.obs


def test_write_anndata_var_column(
    mdata: MuData,
    write_func: Callable[Concatenate[str | Path, MuData, ...], None],
    read_func: Callable[[str | Path], MuData],
    filepath: Path,
) -> None:
    write_func(filepath, mdata)

    mdata["mod1"].var["assert-bool"] = np.arange(mdata["mod1"].n_vars)
    mdata["mod1"].var["unwritten"] = 1
    mdata["mod2"].var["unwritten"] = 1
    mdata.var["unwritten"] = 1

    write_func(filepath, mdata, use=mdA.mod["mod1"].var["assert-bool"])
    mdata_ = read_func(filepath)
    assert (mdata["mod1"].var["assert-bool"] == mdata_["mod1"].var["assert-bool"]).all()
    assert "assert-bool" not in mdata_.var
    assert "unwritten" not in mdata_.var
    assert "unwritten" not in mdata_["mod1"].var
    assert "unwritten" not in mdata_["mod2"].var


def test_write_mods(
    mdata: MuData,
    write_func: Callable[Concatenate[str | Path, MuData, ...], None],
    read_func: Callable[[str | Path], MuData],
    filepath: Path,
    rng: np.random.Generator,
) -> None:
    write_func(filepath, mdata)

    mdata.obs["unwritten"] = 1
    mdata.mod["mod3"] = AnnData(
        X=rng.normal(size=(mdata.n_obs, 10)),
        obs=pd.DataFrame(index=mdata.obs_names),
        var=pd.DataFrame(index=[f"mod3_var_{i}" for i in range(10)]),
    )

    with pytest.warns(RuntimeWarning, match="do not correspond to those already on disk"):
        write_func(filepath, mdata, use=mdA.mod)
    mdata_ = read_func(filepath)
    assert "unwritten" not in mdata_.obs
    for modname, mod in mdata.mod.items():
        assert_equal(mod, mdata_.mod[modname])

    mdata.mod["mod2"].obs["new-col"] = rng.normal(size=mdata.mod["mod2"].n_obs)
    mdata.mod["mod3"].obs["unwritten"] = rng.normal(size=mdata.mod["mod3"].n_obs)
    write_func(filepath, mdata, use=mdA.mod["mod2"])
    mdata_ = read_func(filepath)
    assert (mdata.mod["mod2"].obs["new-col"] == mdata.mod["mod2"].obs["new-col"]).all()
    assert "unwritten" not in mdata_.mod["mod3"].obs

    mdata.update()
    with pytest.raises(ValueError, match="var_names in the store do not match"):
        write_func(filepath, mdata, use=mdA.mod)


def test_write_anndata_obsp(
    mdata: MuData,
    write_func: Callable[Concatenate[str | Path, MuData, ...], None],
    read_func: Callable[[str | Path], MuData],
    filepath: Path,
    rng: np.random.Generator,
) -> None:
    write_func(filepath, mdata)

    mdata.mod["mod1"].obsp["test"] = rng.poisson(size=(mdata.mod["mod1"].n_obs, mdata.mod["mod1"].n_obs))
    mdata.mod["mod2"].obsp["test"] = rng.poisson(size=(mdata.mod["mod2"].n_obs, mdata.mod["mod2"].n_obs))
    write_func(filepath, mdata, use=mdA.mod["mod1"].obsp["test"])
    mdata_ = read_func(filepath)
    assert_equal(mdata.mod["mod1"], mdata_.mod["mod1"])
    assert "test" not in mdata_.mod["mod2"].obsp

    write_func(filepath, mdata, use=mdA.mod["mod2"].obsp)
    mdata_ = read_func(filepath)
    assert_equal(mdata.mod["mod1"], mdata_.mod["mod1"])


def test_write_errors(
    mdata: MuData,
    write_func: Callable[Concatenate[str | Path, MuData, ...], None],
    read_func: Callable[[str | Path], MuData],
    filepath: Path,
    rng: np.random.Generator,
) -> None:
    write_func(filepath, mdata)

    mdata.mod["mod3"] = AnnData(
        X=rng.normal(size=(mdata.n_obs, 10)),
        obs=pd.DataFrame(index=mdata.obs_names),
        var=pd.DataFrame(index=[f"mod3_var_{i}" for i in range(10)]),
    )

    with pytest.raises(ValueError, match="is not present on disk, cannot write part of it"):
        write_func(filepath, mdata, use=mdA.mod["mod3"].obs)
    with pytest.raises(ValueError, match="is not present on disk, cannot write part of it"):
        write_func(filepath, mdata, use=mdA.mod["mod3"].obs["testcol"])

    with pytest.warns(UserWarning, match="obs_names are not unique"):
        mdata_ = MuData(mdata.mod, axis=1)
    with pytest.raises(ValueError, match="axis in the store does not match"):
        write_func(filepath, mdata_, use=mdA.mod)
