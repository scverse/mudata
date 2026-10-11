from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata.tests.helpers import assert_equal
from packaging.version import Version

import mudata as md
from mudata.tests.reference import generate_reference

ARCHIVE_PATH = Path(__file__).parent / "data" / "archives"


@pytest.fixture(params=tuple(sorted(ARCHIVE_PATH.glob("v*/*.h5mu"))), ids=lambda x: f"{x.parent.name}-{x.stem}")
def archive_file(request: pytest.FixtureRequest) -> Path:
    return request.param


@pytest.fixture
def reference(archive_file: Path) -> md.MuData:
    axis = int(archive_file.stem[-1])
    mdata = generate_reference(axis)
    if Version(archive_file.parent.name) < Version("0.4"):
        mdata.pull_obs(unique=False, nonunique=False)
        if axis == 0:
            mdata.pull_var(unique=False, nonunique=False)
    mdata.strings_to_categoricals()
    if Version(archive_file.parent.name) >= Version("0.4"):
        # Recent archives store StringDtype indices; AnnData < 0.13 ignores their na-value="NaN".
        dtype = pd.StringDtype(na_value=np.nan) if Version(version("anndata")) >= Version("0.13") else pd.StringDtype()
        for obj in (mdata, *mdata.mod.values()):
            obj.obs.index = obj.obs.index.astype(dtype)
            obj.var.index = obj.var.index.astype(dtype)
        mdata["mod1"].raw.var.index = mdata["mod1"].raw.var.index.astype(dtype)
    return mdata


@pytest.mark.filterwarnings("ignore::anndata.OldFormatWarning")
def test_backwards_compat_h5mu(tmp_path: Path, archive_file: Path, reference: md.MuData):
    from_h5mu = md.read_h5mu(archive_file)
    assert_equal(from_h5mu, reference, exact=True)

    with ad.settings.override(allow_write_nullable_strings=True):
        md.write_h5mu(new_path := tmp_path / archive_file.name, from_h5mu)
    assert_equal(from_h5mu, md.read_h5mu(new_path), exact=True)


@pytest.mark.filterwarnings("ignore::anndata.OldFormatWarning")
def test_backwards_compat_zarr(tmp_path: Path, archive_file: Path, reference: md.MuData):
    fname_zarr = archive_file.with_suffix(".zarr.zip")
    if Version(zarr.__version__).major < 3:
        with ZipFile(fname_zarr) as archive:
            if "zarr.json" in archive.namelist():
                pytest.skip("Zarr format 3 archives require zarr-python >= 3")

    with zarr.storage.ZipStore(fname_zarr, mode="r") as store:
        from_zarr = md.read_zarr(store)
    assert_equal(from_zarr, reference, exact=True)
    assert_equal(from_zarr, md.read_h5mu(archive_file), exact=True)

    # A .zip suffix selects ZipStore in Zarr 2; use a directory for the round trip.
    with ad.settings.override(allow_write_nullable_strings=True):
        md.write_zarr(new_path := tmp_path / f"{archive_file.stem}.zarr", from_zarr)
    assert_equal(from_zarr, md.read_zarr(new_path), exact=True)
