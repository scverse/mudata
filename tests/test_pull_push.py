import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from mudata import MuData


@pytest.fixture(params=(0, 1))
def axis(request):
    return request.param


@pytest.fixture
def attr(axis):
    return "obs" if axis == 0 else "var"


@pytest.fixture
def oattr(axis):
    return "var" if axis == 0 else "obs"


@pytest.fixture(params=("joint", "disjoint"))
def n(request):
    return request.param


@pytest.fixture(params=(True, False))
def unique(request):
    return request.param


@pytest.fixture
def modalities(rng, axis, attr, n, unique):
    n_mod = 3
    mods = {}
    for i in range(n_mod):
        i1 = i + 1
        m = f"mod{i1}"
        mods[m] = AnnData(X=rng.normal(size=1000 * i1).reshape(-1, 10 * i1))
        mods[m].obs["mod"] = m
        mods[m].var["mod"] = m
        mods[m].obs["assert-bool"] = True
        mods[m].obs[f"assert-boolean-{m}"] = False
        mods[m].var["assert-bool"] = True
        mods[m].var[f"assert-boolean-{m}"] = False
        mods[m].obs["min_count"] = mods[m].X.min(axis=1)

        if unique:
            setattr(mods[m], f"{attr}_names", [f"{m}_{attr}{j}" for j in range(mods[m].shape[axis])])

        # common column
        for axis_, attr_ in enumerate(("obs", "var")):
            df = getattr(mods[m], attr_)
            df["common_col"] = rng.integers(0, int(1e6), size=mods[m].shape[axis_])

            df["dtype-int-common"] = np.arange(mods[m].shape[axis_])
            df["dtype-float-common"] = np.linspace(0, 1, mods[m].shape[axis_], dtype=np.float32)
            df["dtype-bool-common"] = rng.choice(1, mods[m].shape[axis_]).astype(bool)
            df["dtype-string-common"] = rng.choice(["a", "b", "c"], size=mods[m].shape[axis_])

            if i != 0:
                # non-unique column missing from mod1
                df["nonunique_col"] = np.arange(mods[m].shape[axis_])
                df["dtype-int-nonunique"] = np.arange(mods[m].shape[axis_])
                df["dtype-float-nonunique"] = np.linspace(0, 1, mods[m].shape[axis_], dtype=np.float32)
                df["dtype-bool-nonunique"] = rng.choice(1, mods[m].shape[axis_]).astype(bool)
                df["dtype-string-nonunique"] = rng.choice(["a", "b", "c"], size=mods[m].shape[axis_])
            else:
                # mod1-specific column
                df["unique_col"] = True
                df["dtype-int-unique"] = np.arange(mods[m].shape[axis_])
                df["dtype-float-unique"] = np.linspace(0, 1, mods[m].shape[axis_], dtype=np.float32)
                df["dtype-bool-unique"] = rng.choice(1, mods[m].shape[axis_]).astype(bool)
                df["dtype-string-unique"] = rng.choice(["a", "b", "c"], size=mods[m].shape[axis_])

    if n:
        if n == "disjoint":
            mod2_which = rng.choice(
                getattr(mods["mod2"], f"{attr}_names"), size=mods["mod2"].shape[axis] // 2, replace=False
            )
            idx = (mod2_which, slice(None)) if axis == 0 else (slice(None), mod2_which)
            mods["mod2"] = mods["mod2"][idx].copy()

    return mods


def assert_dtypes(df, suffix, prefix=""):
    assert pd.api.types.is_integer_dtype(df[f"{prefix}dtype-int-{suffix}"])
    assert pd.api.types.is_float_dtype(df[f"{prefix}dtype-float-{suffix}"])
    assert pd.api.types.is_bool_dtype(df[f"{prefix}dtype-bool-{suffix}"])
    assert pd.api.types.is_string_dtype(df[f"{prefix}dtype-string-{suffix}"]) or df[f"{prefix}dtype-string-{suffix}"].dtype == object


def test_pull_oattr(modalities, axis, oattr):
    """
    Pulling var annotations shouldn't depend on var_names uniqueness.
    """
    mdata = MuData(modalities, axis=axis)
    pull_func = lambda *args, **kwargs: getattr(mdata, f"pull_{oattr}")(*args, **kwargs) or getattr(mdata, oattr)
    reset_df = lambda: setattr(mdata, oattr, getattr(mdata, oattr).iloc[:, []])

    df = pull_func()
    assert "mod" in df.columns
    assert "assert-bool" in df.columns
    assert df["assert-bool"].dtype == bool
    for dtype in ("int", "float", "bool", "string"):
        assert f"dtype-{dtype}-common" in df.columns

        assert f"dtype-{dtype}-nonunique" not in df.columns
        assert f"dtype-{dtype}-unique" not in df.columns
    assert_dtypes(df, "common")

    for m, mod in modalities.items():
        assert df[f"{m}:assert-boolean-{m}"].dtype == "boolean"
        # Annotations are correct
        assert all(df.loc[getattr(mdata, f"{oattr}map")[m] > 0, "mod"] == m)
        # Columns are intact in individual modalities
        mdf = getattr(mod, oattr)
        assert "mod" in mdf.columns
        assert all(mdf["mod"] == m)

    reset_df()

    # Pull a common column
    df = pull_func(columns=["common_col"])
    assert "common_col" in df.columns
    assert (~pd.isnull(df.common_col)).sum() == mdata.shape[1 - axis]
    reset_df()

    # Pull a common column from one modality
    df = pull_func(columns=["common_col"], mods=["mod2"])
    assert "mod2:common_col" in df.columns
    assert (~pd.isnull(df["mod2:common_col"])).sum() == mdata["mod2"].shape[1 - axis]
    reset_df()

    # do not pull unique columns
    df = pull_func(common=True, nonunique=True, unique=False)
    assert "mod1:unique_col" not in df.columns
    assert "common_col" in df.columns
    assert "nonunique_col" not in df.columns

    assert_dtypes(df, "common")
    assert_dtypes(df, "nonunique", "mod2:")
    assert_dtypes(df, "nonunique", "mod3:")
    reset_df()

    # only pull a unique column
    df = pull_func(common=False, nonunique=False, unique=True)
    assert "mod1:unique_col" in df.columns
    assert len(df.columns) == 8
    assert_dtypes(df, "unique", "mod1:")
    reset_df()

    # pull non-unique but do not join
    df = pull_func(common=False, unique=False)
    assert "nonunique_col" not in df.columns
    assert len(df.columns) == (mdata.n_mod - 1) * 5
    assert_dtypes(df, "nonunique", "mod2:")
    assert_dtypes(df, "nonunique", "mod3:")
    reset_df()

    # pull non-unique and join them
    df = pull_func(common=False, unique=False, join_nonunique=True)
    assert "nonunique_col" in df.columns
    assert len(df.columns) == 5
    assert_dtypes(df, "nonunique")
    reset_df()

    # pull unique and do not prefix
    df = pull_func(common=False, nonunique=False, unique=True, prefix_unique=False)
    assert "mod1:unique_col" not in df.columns
    assert "unique_col" in df.columns
    assert len(df.columns) == 8
    assert_dtypes(df, "unique")


def test_pull_attr_simple(modalities, axis, attr):
    """
    Pulling obs annotations.
    """
    mdata = MuData(modalities, axis=axis)
    pull_func = getattr(mdata, f"pull_{attr}")

    pull_func()
    df = getattr(mdata, attr)

    # pulling should work
    for m, mod in mdata.mod.items():
        assert f"{m}:mod" in df.columns
        assert f"{m}:common_col" in df.columns

        for dtype in ("int", "float", "bool", "string"):
            assert f"{m}:dtype-{dtype}-common" in df.columns

        assert_dtypes(df, "common", f"{m}:")

        modmap = getattr(mdata, f"{attr}map")[m].ravel()
        mask = modmap > 0
        assert (
            df[f"{m}:common_col"][mask].to_numpy() == getattr(mod, attr)["common_col"].to_numpy()[modmap[mask] - 1]
        ).all()

    # join_common shouldn't work
    with pytest.raises(ValueError, match=f"shared {attr}_names"):
        pull_func(join_common=True)

    # join_nonunique shouldn't work
    with pytest.raises(ValueError, match=f"shared {attr}_names"):
        pull_func(join_nonunique=True)


def test_push_oattr_simple(rng, modalities, axis, oattr):
    """
    Pushing var annotations.
    """
    mdata = MuData(modalities, axis=axis)
    df = getattr(mdata, oattr)

    df["dtype-int-pushed"] = np.arange(mdata.shape[1 - axis])
    df["dtype-float-pushed"] = np.linspace(0, 1, mdata.shape[1 - axis], dtype=np.float32)
    df["dtype-bool-pushed"] = rng.choice(1, mdata.shape[1 - axis]).astype(bool)
    df["dtype-string-pushed"] = rng.choice(["a", "b", "c"], size=mdata.shape[1 - axis])
    df["mod2:mod2_dtype-int-pushed"] = np.arange(mdata.shape[1 - axis])
    df["mod2:mod2_dtype-float-pushed"] = np.linspace(0, 1, mdata.shape[1 - axis], dtype=np.float32)
    df["mod2:mod2_dtype-bool-pushed"] = rng.choice(1, mdata.shape[1 - axis]).astype(bool)
    df["mod2:mod2_dtype-string-pushed"] = rng.choice(["a", "b", "c"], size=mdata.shape[1 - axis])
    getattr(mdata, f"push_{oattr}")()

    # pushing should work
    for modname, mod in mdata.mod.items():
        mdf = getattr(mod, oattr)
        assert_dtypes(mdf, "pushed")

        map = getattr(mdata, f"{oattr}map")[modname].ravel()
        mask = map > 0
        assert (df["dtype-int-pushed"][mask] == mdf["dtype-int-pushed"].iloc[map[mask] - 1]).all()

    assert_dtypes(getattr(mdata["mod2"], oattr), "pushed", "mod2_")
    map = getattr(mdata, f"{oattr}map")["mod2"].ravel()
    mask = map > 0
    assert (
        df["mod2:mod2_dtype-int-pushed"][mask]
        == getattr(mdata["mod2"], oattr)["mod2_dtype-int-pushed"].iloc[map[mask] - 1]
    ).all()


def test_push_axis_simple(rng, modalities, axis, attr):
    """
    Pushing obs annotations.
    """
    mdata = MuData(modalities, axis=axis)
    df = getattr(mdata, attr)

    df["pushed"] = rng.integers(0, int(1e6), size=mdata.shape[axis])
    df["mod2:mod2_pushed"] = rng.integers(0, int(1e6), size=mdata.shape[axis])
    getattr(mdata, f"push_{attr}")()

    # pushing should work
    for modname, mod in mdata.mod.items():
        mdf = getattr(mod, attr)
        assert "pushed" in mdf.columns

        map = getattr(mdata, f"{attr}map")[modname].ravel()
        mask = map > 0
        assert (df["pushed"][mask] == mdf["pushed"].iloc[map[mask] - 1]).all()

    assert "mod2_pushed" in getattr(mdata["mod2"], attr).columns
    map = getattr(mdata, f"{attr}map")["mod2"].ravel()
    mask = map > 0
    assert (df["mod2:mod2_pushed"][mask] == getattr(mdata["mod2"], attr)["mod2_pushed"].iloc[map[mask] - 1]).all()
