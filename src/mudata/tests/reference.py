#!/usr/bin/env python3

from typing import Literal

import numpy as np
import pandas as pd
import zarr
from anndata import AnnData

from mudata import MuData, write_h5mu, write_zarr


def generate_reference(axis: Literal[0, 1]) -> MuData:
    """Generate a reference MuData object to test backwards compatibility."""
    mod1 = AnnData(
        np.arange(0, 200, 0.1, dtype=np.float32).reshape(-1, 20), obs=pd.DataFrame(index=[str(i) for i in range(100)])
    )
    mod2 = AnnData(
        np.arange(101, 3101, 1, dtype=np.float32).reshape(-1, 30),
        obs=pd.DataFrame(index=[str(i) for i in range(99, -1, -1)]),
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
        setattr(mod, f"{attr}_names", [f"{attr}_{i}" for i in range(mod.shape[axis])])
        setattr(mod, f"{oattr}_names", [f"{modname}_{oattr}_{i}" for i in range(mod.shape[1 - axis])])
    mod1.obs.index.name = "fizz"
    mod2.var.index.name = "buzz"
    mdata = MuData(mods, axis=axis)
    mdata.obs["arange"] = np.arange(mdata.n_obs)
    return mdata


if __name__ == "__main__":
    for axis in (0, 1):
        mdata = generate_reference(axis)
        write_h5mu(f"mdata_axis{axis}.h5mu", mdata, compression="gzip", compression_opts=9)
        write_zarr(zarr.storage.ZipStore(f"mdata_axis{axis}.zarr.zip", mode="w"), mdata)
