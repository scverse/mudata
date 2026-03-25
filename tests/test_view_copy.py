from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import mudata as md


@pytest.fixture
def mdata_with_obsp(mdata: md.MuData):
    distances = sparse.random(mdata.n_obs, mdata.n_obs, density=0.2, random_state=42)
    distances = sparse.triu(distances)
    distances = distances + distances.T

    connectivities = sparse.random(mdata.n_obs, mdata.n_obs, density=0.1, random_state=42)
    connectivities = sparse.triu(connectivities)
    connectivities = connectivities + connectivities.T

    mdata.obsp["distances"] = distances
    mdata.obsp["connectivities"] = connectivities

    varp_matrix = sparse.random(mdata.n_var, mdata.n_var, density=0.05, random_state=42)
    varp_matrix = sparse.triu(varp_matrix)
    varp_matrix = varp_matrix + varp_matrix.T

    mdata.varp["correlations"] = varp_matrix

    return mdata


def test_copy(mdata: md.MuData):
    mdata_copy = mdata.copy()
    assert mdata.shape == mdata_copy.shape
    assert np.array_equal(mdata.obs_names.values, mdata_copy.obs_names.values)
    assert np.array_equal(mdata.var_names.values, mdata_copy.var_names.values)
    assert np.array_equal(mdata.obs.columns.values, mdata_copy.obs.columns.values)
    assert np.array_equal(mdata.var.columns.values, mdata_copy.var.columns.values)


def test_view_attributes(mdata: md.MuData):
    n, d = mdata.n_obs, mdata.n_var
    # Populate attributes
    mdata.uns["uns_key"] = {"key": "value"}
    mdata.obs["obs_column"] = False
    mdata.var["var_column"] = False
    mdata.obsm["obsm_key"] = np.arange(n).reshape(-1, 1)
    mdata.varm["varm_key"] = np.arange(d).reshape(-1, 1)
    mdata.obsp["obsp_key"] = np.arange(n * n).reshape(n, n)
    mdata.varp["varp_key"] = np.arange(d * d).reshape(d, d)

    view_n_obs = 7
    mdata_view = mdata[list(range(view_n_obs)), :]
    assert mdata_view.shape == (view_n_obs, mdata.n_var)
    assert len(mdata_view.mod) == len(mdata.mod)
    # AnnData/MuData interface
    for attr in "obs", "var", "obsm", "varm", "obsp", "varp", "uns":
        assert hasattr(mdata_view, attr)
        assert list(getattr(mdata_view, attr).keys()) == list(getattr(mdata, attr).keys())
    # MuData-specific interface
    for attr in "mod", "axis", "obsmap", "varmap":
        assert hasattr(mdata_view, attr)
    assert mdata_view.axis == mdata.axis


def test_view_copy(mdata: md.MuData):
    view_n_obs = 5
    mdata_view = mdata[list(range(view_n_obs)), :]
    assert mdata_view.is_view
    assert mdata_view.n_obs == view_n_obs
    mdata_copy = mdata_view.copy()
    assert not mdata_copy.is_view
    assert mdata_copy.n_obs == view_n_obs


def test_view_view(mdata: md.MuData):
    view_n_obs = 5
    mdata_view = mdata[list(range(view_n_obs)), :]
    assert mdata_view.is_view
    assert mdata_view.n_obs == view_n_obs

    for modname, mod in mdata_view.mod.items():
        assert mdata_view.obsmap[modname].max() == mod.n_obs
        idx = mdata_view.obsmap[modname]
        assert np.all(mdata_view.obs_names[idx > 0] == mod.obs_names[idx[idx > 0] - 1])

    view_view_n_obs = 2
    mdata_view_view = mdata_view[list(range(view_view_n_obs)), :]
    assert mdata_view_view.is_view
    assert mdata_view_view.n_obs == view_view_n_obs

    for modname, mod in mdata_view_view.mod.items():
        ref_obsmap = mdata_view.obsmap[modname][:view_view_n_obs]
        ref_obsmap = ref_obsmap[ref_obsmap > 0] - 1
        assert (mod.obs_names == mdata_view[modname].obs_names[ref_obsmap]).all()
        assert (mod.var_names == mdata_view[modname].var_names).all()

    # test reordering
    mdata_view_view = mdata_view[:, :]
    for modname, mod in mdata_view_view.mod.items():
        assert (mod.obs_names == mdata_view[modname].obs_names).all()
        assert (mod.var_names == mdata_view[modname].var_names).all()


def test_backed_copy(mdata: md.MuData, filepath_h5mu: str | Path, filepath2_h5mu: str | Path):
    mdata.write(filepath_h5mu)
    mdata_b = md.read_h5mu(filepath_h5mu, backed="r")
    assert mdata_b.n_obs == mdata.n_obs
    mdata_b_copy = mdata_b.copy(filepath2_h5mu)
    assert mdata_b_copy.file._filename.name == Path(filepath2_h5mu).name


def test_obsp_slicing(mdata_with_obsp: md.MuData):
    """Test that obsp matrices are correctly sliced when subsetting a MuData object."""
    orig_n_obs = mdata_with_obsp.n_obs

    # Check initial shapes
    assert mdata_with_obsp.obsp["distances"].shape == (orig_n_obs, orig_n_obs)
    assert mdata_with_obsp.obsp["connectivities"].shape == (orig_n_obs, orig_n_obs)

    # Slice a subset of cells
    n_obs_subset = 50
    random_indices = np.random.choice(mdata_with_obsp.obs_names, size=n_obs_subset, replace=False)

    # Create a slice view
    mdata_slice = mdata_with_obsp[random_indices]

    # Check that the sliced obsp matrices have correct shape in the view
    assert mdata_slice.obsp["distances"].shape == (n_obs_subset, n_obs_subset), (
        f"Expected shape in view: {(n_obs_subset, orig_n_obs)}, got: {mdata_slice.obsp['distances'].shape}"
    )
    assert mdata_slice.obsp["connectivities"].shape == (n_obs_subset, n_obs_subset), (
        f"Expected shape in view: {(n_obs_subset, orig_n_obs)}, got: {mdata_slice.obsp['connectivities'].shape}"
    )

    # Make a copy of the sliced MuData object
    mdata_copy = mdata_slice.copy()
    # Check shapes after copy - these should be (n_obs_subset, n_obs_subset) if correctly copied
    assert mdata_copy.obsp["distances"].shape == (n_obs_subset, n_obs_subset), (
        f"Expected shape after copy: {(n_obs_subset, n_obs_subset)}, got: {mdata_copy.obsp['distances'].shape}"
    )
    assert mdata_copy.obsp["connectivities"].shape == (n_obs_subset, n_obs_subset), (
        f"Expected shape after copy: {(n_obs_subset, n_obs_subset)}, got: {mdata_copy.obsp['connectivities'].shape}"
    )


def test_varp_slicing(mdata_with_obsp: md.MuData):
    """Test that varp matrices are correctly sliced when subsetting a MuData object."""
    orig_n_var = mdata_with_obsp.n_var

    # Check initial shape
    assert mdata_with_obsp.varp["correlations"].shape == (orig_n_var, orig_n_var)

    # Slice a subset of variables
    n_var_subset = 15
    all_var_names = mdata_with_obsp.var_names
    random_var_indices = np.random.choice(all_var_names, size=n_var_subset, replace=False)

    # Create a slice view
    mdata_slice = mdata_with_obsp[:, random_var_indices]

    # Check that the sliced varp matrix has correct shape in the view
    assert mdata_slice.varp["correlations"].shape == (n_var_subset, n_var_subset), (
        f"Expected shape in view: {(n_var_subset, orig_n_var)}, got: {mdata_slice.varp['correlations'].shape}"
    )

    # Copy the sliced MuData object
    mdata_copy = mdata_slice.copy()
    # Check shapes after copy
    assert mdata_copy.varp["correlations"].shape == (n_var_subset, n_var_subset), (
        f"Expected shape after copy: {(n_var_subset, n_var_subset)}, got: {mdata_copy.varp['correlations'].shape}"
    )


def _test_view_after_setattr(mdata: md.MuData, mdata_ref: md.MuData, skip: Sequence[str] = ()):
    if isinstance(skip, str):
        skip = (skip,)

    assert not mdata.is_view
    assert mdata.mod.keys() == mdata_ref.mod.keys()
    for attr in ("obsm", "varm", "obsp", "varp", "obsmap", "varmap", "uns"):
        if attr not in skip:
            assert getattr(mdata, attr).keys() == getattr(mdata_ref, attr).keys()
    for attr in ("obs", "var"):
        if attr not in skip:
            assert (getattr(mdata, attr).columns == getattr(mdata_ref, attr).columns).all()
    for mod in mdata.mod.values():
        assert not mod.is_view

    for k, v in mdata.obsm.items():
        if k in mdata_ref.obsm:
            assert v.shape[1] == mdata_ref.obsm[k].shape[1]
            if isinstance(v, pd.DataFrame):
                assert (v.columns == mdata_ref.obsm[k].columns).all()
    for k, v in mdata.varm.items():
        if k in mdata_ref.varm:
            assert v.shape[1] == mdata_ref.obsm[k].shape[1]
            if isinstance(v, pd.DataFrame):
                assert (v.columns == mdata_ref.varm[k].columns).all()


def test_view_setattr(mdata_with_obsp: md.MuData):
    view = mdata_with_obsp[:42, :]
    view.obs_names = [f"foo_{i}" for i in range(len(view))]
    _test_view_after_setattr(view, mdata_with_obsp)

    view = mdata_with_obsp[:42, :]
    view.uns["foo"] = 42
    _test_view_after_setattr(view, mdata_with_obsp, "uns")

    view = mdata_with_obsp[:42, :]
    view.obsp["distances"][1, 2] = 42
    _test_view_after_setattr(view, mdata_with_obsp)

    view = mdata_with_obsp[:42, :]
    view.varp["correlations"][1, 2] = 42
    _test_view_after_setattr(view, mdata_with_obsp)

    view = mdata_with_obsp[:42, :]
    view.obs["foo"] = 42
    _test_view_after_setattr(view, mdata_with_obsp, "obs")

    view = mdata_with_obsp[:42, :]
    view.var = pd.DataFrame(index=[f"foo_{i}" for i in range(view.shape[1])])
    _test_view_after_setattr(view, mdata_with_obsp, "var")

    view = mdata_with_obsp[:42, :]
    view.obsm["foo"] = np.arange(2 * len(view)).reshape(-1, 2)
    _test_view_after_setattr(view, mdata_with_obsp, "obsm")

    view = mdata_with_obsp[:42, :]
    view.varm["foo"] = np.arange(2 * view.shape[1]).reshape(-1, 2)
    _test_view_after_setattr(view, mdata_with_obsp, "varm")

    for attr in ("obsm", "varm", "obsp", "varp", "uns"):
        view = mdata_with_obsp[:42, :]
        delattr(view, attr)
        _test_view_after_setattr(view, mdata_with_obsp, attr)
