from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from mudata import MuData


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
def mdata(rng: np.random.Generator, request: pytest.FixtureRequest):
    axis = getattr(request, "param", 0)
    mod1 = AnnData(
        np.arange(0, 200, 0.1).reshape(-1, 20), obs=pd.DataFrame(index=rng.choice(150, size=100, replace=False))
    )
    mod2 = AnnData(
        np.arange(101, 3101, 1).reshape(-1, 30), obs=pd.DataFrame(index=rng.choice(150, size=100, replace=False))
    )
    mod1.var["assert-bool"] = True
    mod2.var["assert-bool"] = False
    mod1.var["assert-boolean-1"] = True
    mod2.var["assert-boolean-2"] = False
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
    mdata = MuData(mods, axis=axis)
    mdata.obs["arange"] = np.arange(mdata.n_obs)
    return mdata


@pytest.fixture
def mdata_nouniqueobs(mdata: MuData):
    return mdata[mdata["mod1"].obs_names.intersection(mdata["mod2"].obs_names), :]
