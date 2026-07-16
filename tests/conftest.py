from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from anndata.tests.helpers import assert_equal

from mudata import MuData


def unify_string_dtypes(x):
    match x.dtype:
        case "str":
            return x.astype("string")
        case pd.CategoricalDtype(categories=cats) if cats.dtype == "str":
            return x.cat.set_categories(x.cat.categories.astype("string"))
        case _:
            return x


@assert_equal.register(MuData)
def assert_mdata_equal(a: MuData, b: object, *, exact: bool = False):
    assert isinstance(b, MuData)
    assert a.axis == b.axis
    assert a.mod.keys() == b.mod.keys()

    assert_equal(a.obs_names, b.obs_names, exact=exact, elem_name="obs_names")
    assert_equal(a.var_names, b.var_names, exact=exact, elem_name="var_names")

    for attr in ("obs", "var"):
        assert_equal(getattr(a, attr).transform(unify_string_dtypes), getattr(b, attr).transform(unify_string_dtypes))

    for attr in ("obsm", "varm", "obsp", "varp", "obsmap", "varmap", "uns"):
        assert_equal(getattr(a, attr), getattr(b, attr), exact=exact, elem_name=attr)

    for m, mod in a.mod.items():
        assert_equal(mod, b.mod[m], exact=exact, elem_name=f"mod/{m}")


@pytest.fixture
def filepath_h5mu(tmp_path: Path) -> Path:
    return tmp_path / "testA.h5mu"


@pytest.fixture
def filepath2_h5mu(tmp_path: Path) -> Path:
    return tmp_path / "testB.h5mu"


@pytest.fixture
def filepath_zarr(tmp_path: Path) -> Path:
    return tmp_path / "testA.zarr"


@pytest.fixture
def filepath2_zarr(tmp_path: Path) -> Path:
    return tmp_path / "testB.zarr"


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def mdata(rng: np.random.Generator, request: pytest.FixtureRequest) -> MuData:
    axis = getattr(request, "param", 0)
    mod1 = AnnData(
        np.arange(0, 200, 0.1).reshape(-1, 20),
        obs=pd.DataFrame(index=rng.choice(150, size=100, replace=False).astype(str)),
    )
    mod2 = AnnData(
        np.arange(101, 3101, 1).reshape(-1, 30),
        obs=pd.DataFrame(index=rng.choice(150, size=100, replace=False).astype(str)),
    )
    mod1.var["assert-bool"] = True
    mod2.var["assert-bool"] = False
    mod1.var["assert-boolean-1"] = True
    mod2.var["assert-boolean-2"] = False

    mod1.raw = mod1[:, :10].copy()
    mods = {"mod2": mod2, "mod1": mod1}

    attr = "obs" if axis == 0 else "var"
    oattr = "var" if axis == 0 else "obs"
    for modname, mod in mods.items():
        mod.var["feature_type"] = modname
        setattr(
            mod,
            f"{attr}_names",
            [f"{attr}_{i}" for i in rng.choice(mod.shape[axis], size=mod.shape[axis], replace=False)],
        )
        setattr(mod, f"{oattr}_names", [f"{modname}_{oattr}_{i}" for i in range(mod.shape[1 - axis])])
    mod1.obs.index.name = "fizz"
    mod2.var.index.name = "buzz"
    mdata = MuData(mods, axis=axis)
    mdata.obs["arange"] = np.arange(mdata.n_obs)
    return mdata


@pytest.fixture
def mdata_nouniqueobs(mdata: MuData):
    return mdata[mdata["mod1"].obs_names.intersection(mdata["mod2"].obs_names), :]
