from pathlib import Path

import pytest
import zarr
from anndata.tests.helpers import assert_equal
from packaging.version import Version

import mudata as md
from mudata.tests.reference import generate_reference

ARCHIVE_PATH = Path(__file__).parent / "data" / "archives"


@pytest.fixture(params=tuple(ARCHIVE_PATH.glob("v*")), ids=lambda x: x.name)
def archive_dir(request: pytest.FixtureRequest):
    return request.param


@pytest.mark.filterwarnings("ignore::anndata.OldFormatWarning")
def test_backwards_compat_files(subtests: pytest.Subtests, tmp_path: Path, archive_dir: Path):
    for fname in archive_dir.glob("*.h5mu"):
        axis = int(fname.stem[-1])
        reference = generate_reference(axis)
        if Version(archive_dir.stem) < Version("0.4"):
            reference.pull_obs(unique=False, nonunique=False)
            if axis == 0:
                reference.pull_var(unique=False, nonunique=False)
        reference.strings_to_categoricals()
        with subtests.test(msg=fname.stem):
            fname_zarr = fname.parent / f"{fname.stem}.zarr.zip"
            from_h5mu = md.read_h5mu(fname)
            from_zarr = md.read_zarr(zarr.storage.ZipStore(fname_zarr))

            assert_equal(from_h5mu, from_zarr, exact=True)

            md.write_h5mu(new_h5mu_path := tmp_path / fname.name, from_h5mu)
            assert_equal(from_h5mu, md.read_h5mu(new_h5mu_path), exact=True)

            md.write_zarr(new_zarr_path := tmp_path / fname_zarr.name, from_zarr)
            assert_equal(from_zarr, md.read_zarr(new_zarr_path), exact=True)

            assert_equal(from_h5mu, reference, exact=True)
