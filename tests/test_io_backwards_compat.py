from pathlib import Path

import pandas as pd
import pytest
import zarr
from anndata.tests.helpers import assert_equal
from packaging.version import Version

import mudata as md
from mudata.tests.reference import generate_reference


def unify_string_dtypes(x):
    match x.dtype:
        case "str":
            return x.astype("string")
        case pd.CategoricalDtype(categories=cats) if cats.dtype == "str":
            return x.cat.set_categories(x.cat.categories.astype("string"))
        case _:
            return x


@assert_equal.register(md.MuData)
def assert_mdata_equal(a: md.MuData, b: object, *, exact: bool = False):
    assert isinstance(b, md.MuData)
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


ARCHIVE_PATH = Path(__file__).parent / "data" / "archives"

ARCHIVES = tuple(ARCHIVE_PATH.glob("v*"))


@pytest.fixture(params=ARCHIVES, ids=lambda x: x.name)
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
