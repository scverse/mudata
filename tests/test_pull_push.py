from collections.abc import Callable
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from mudata import MuData, set_options

Axis: TypeAlias = Literal[0, 1]
AxisAttr: TypeAlias = Literal["obs", "var"]


@pytest.fixture(params=(0, 1))
def axis(request) -> Axis:
    return request.param


@pytest.fixture
def attr(axis: Axis) -> AxisAttr:
    return "obs" if axis == 0 else "var"


@pytest.fixture
def oattr(axis: Axis) -> AxisAttr:
    return "var" if axis == 0 else "obs"


@pytest.fixture(params=("joint", "disjoint"))
def n(request) -> Literal["joint", "disjoint"]:
    return request.param


@pytest.fixture(params=(True, False))
def unique(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture
def new_update() -> None:
    set_options(pull_on_update=False)
    yield
    set_options(pull_on_update=None)


@pytest.fixture
def mdata(
    rng: np.random.Generator,
    axis: Axis,
    attr: AxisAttr,
    n: Literal["joint", "disjoint"],
    unique: bool,
    new_update: None,
) -> MuData:
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

    return MuData(mods, axis=axis)


@pytest.fixture
def mdata_for_push(rng: np.random.Generator, mdata: MuData, new_update: None) -> MuData:
    for axis, attr in enumerate(("obs", "var")):
        df = getattr(mdata, attr)

        df["dtype-int-pushed"] = np.arange(mdata.shape[axis])
        df["dtype-float-pushed"] = np.linspace(0, 1, mdata.shape[axis], dtype=np.float32)
        df["dtype-bool-pushed"] = rng.choice(1, mdata.shape[axis]).astype(bool)
        df["dtype-string-pushed"] = rng.choice(["a", "b", "c"], size=mdata.shape[axis])
        df["mod2:mod2_dtype-int-pushed"] = np.arange(mdata.shape[axis])
        df["mod2:mod2_dtype-float-pushed"] = np.linspace(0, 1, mdata.shape[axis], dtype=np.float32)
        df["mod2:mod2_dtype-bool-pushed"] = rng.choice(1, mdata.shape[axis]).astype(bool)
        df["mod2:mod2_dtype-string-pushed"] = rng.choice(["a", "b", "c"], size=mdata.shape[axis])

    return mdata


@pytest.fixture(params=("obs", "var"))
def cattr(request) -> AxisAttr:
    return request.param


@pytest.fixture
def push_func(mdata_for_push: MuData, cattr: AxisAttr) -> Callable[..., None]:
    return getattr(mdata_for_push, f"push_{cattr}")


@pytest.fixture
def push_df(mdata_for_push: MuData, cattr: AxisAttr) -> pd.DataFrame:
    return getattr(mdata_for_push, cattr)


@pytest.fixture
def pull_func_oattr(mdata: MuData, oattr: AxisAttr) -> Callable[..., pd.DataFrame]:
    return lambda *args, **kwargs: getattr(mdata, f"pull_{oattr}")(*args, **kwargs) or getattr(mdata, oattr)


def assert_dtypes(df: pd.DataFrame, suffix: str, prefix: str = ""):
    assert pd.api.types.is_integer_dtype(df[f"{prefix}dtype-int-{suffix}"])
    assert pd.api.types.is_float_dtype(df[f"{prefix}dtype-float-{suffix}"])
    assert pd.api.types.is_bool_dtype(df[f"{prefix}dtype-bool-{suffix}"])
    assert (
        pd.api.types.is_string_dtype(df[f"{prefix}dtype-string-{suffix}"])
        or df[f"{prefix}dtype-string-{suffix}"].dtype == object
    )


def test_raises_on_view(mdata: MuData, attr: AxisAttr):
    getattr(mdata, attr)["foo"] = 42
    view = mdata[:42]

    with pytest.raises(ValueError, match=f"Cannot pull {attr} columns on a view"):
        getattr(view, f"pull_{attr}")("common-col")

    with pytest.raises(ValueError, match=f"Cannot push {attr} columns on a view"):
        getattr(view, f"push_{attr}")("foo")


@pytest.mark.parametrize("drop", (True, False))
def test_pull_oattr_simple(mdata: MuData, oattr: AxisAttr, pull_func_oattr: Callable[..., pd.DataFrame], drop: bool):
    df = pull_func_oattr(drop=drop)
    assert "mod" in df.columns
    assert "common_col" in df.columns
    assert "assert-bool" in df.columns
    assert df["assert-bool"].dtype == bool
    for dtype in ("int", "float", "bool", "string"):
        assert f"dtype-{dtype}-common" in df.columns

        assert f"dtype-{dtype}-nonunique" not in df.columns
        assert f"dtype-{dtype}-unique" not in df.columns
    assert_dtypes(df, "common")

    for m, mod in mdata.mod.items():
        assert f"{m}:assert-boolean-{m}" in df.columns
        assert df[f"{m}:assert-boolean-{m}"].dtype == "boolean"
        # Annotations are correct
        assert all(df.loc[getattr(mdata, f"{oattr}map")[m] > 0, "mod"] == m)

        mdf = getattr(mod, oattr)
        if drop:
            assert "mod" not in mdf.columns
            assert "common_col" not in mdf.columns
        else:
            assert "mod" in mdf.columns
            assert "common_col" in mdf.columns
            assert all(mdf["mod"] == m)
        for dtype in ("int", "float", "bool", "string"):
            if drop:
                assert f"dtype-{dtype}-common" not in mdf.columns
            else:
                assert f"dtype-{dtype}-common" in mdf.columns


def test_pull_oattr_onlydrop(mdata: MuData, oattr: AxisAttr, pull_func_oattr: Callable[..., pd.DataFrame]):
    df = pull_func_oattr(only_drop=True)
    assert "mod" not in df.columns
    assert "common_col" not in df.columns
    assert "assert-bool" not in df.columns
    assert "assert-boolean" not in df.columns
    for dtype in ("int", "float", "bool", "string"):
        assert f"dtype-{dtype}-common" not in df.columns
        assert f"dtype-{dtype}-nonunique" not in df.columns
        assert f"dtype-{dtype}-unique" not in df.columns

    for m, mod in mdata.mod.items():
        assert f"{m}:assert-boolean-{m}" not in df.columns
        mdf = getattr(mod, oattr)
        assert "mod" not in mdf.columns
        assert "common_col" not in mdf.columns

        for dtype in ("int", "float", "bool", "string"):
            assert f"dtype-{dtype}-common" not in mdf.columns
            assert f"dtype-{dtype}-nonunique" not in mdf.columns
            assert f"dtype-{dtype}-unique" not in mdf.columns


@pytest.mark.parametrize("drop", (True, False))
def test_pull_oattr_common(mdata: MuData, oattr: AxisAttr, pull_func_oattr: Callable[..., pd.DataFrame], drop: bool):
    df = pull_func_oattr(columns=["common_col"], drop=drop)
    assert "common_col" in df.columns
    assert (~pd.isnull(df.common_col)).sum() == mdata.shape[1 - mdata.axis]

    if drop:
        for mod in mdata.mod.values():
            assert "common_col" not in getattr(mod, oattr).columns


@pytest.mark.parametrize("drop", (True, False))
def test_pull_oattr_common_mods(
    mdata: MuData, oattr: AxisAttr, pull_func_oattr: Callable[..., pd.DataFrame], drop: bool
):
    df = pull_func_oattr(columns=["common_col"], mods="mod2", drop=drop)
    assert "mod2:common_col" in df.columns
    assert (~pd.isnull(df["mod2:common_col"])).sum() == mdata["mod2"].shape[1 - mdata.axis]

    for modname, mod in mdata.mod.items():
        if drop and modname == "mod2":
            assert "common_col" not in getattr(mod, oattr).columns
        else:
            assert "common_col" in getattr(mod, oattr).columns

    with pytest.raises(ValueError, match="All mods should be present"):
        pull_func_oattr(columns=["common_col"], mods="foo", drop=drop)


@pytest.mark.parametrize("drop", (True, False))
def test_pull_oattr_nounique(mdata: MuData, oattr: AxisAttr, pull_func_oattr: Callable[..., pd.DataFrame], drop: bool):
    df = pull_func_oattr(common=True, nonunique=True, unique=False, drop=drop)
    assert "mod1:unique_col" not in df.columns
    assert "common_col" in df.columns
    assert "nonunique_col" not in df.columns

    assert_dtypes(df, "common")
    assert_dtypes(df, "nonunique", "mod2:")
    assert_dtypes(df, "nonunique", "mod3:")

    for modname, mod in mdata.mod.items():
        mdf = getattr(mod, oattr)
        if drop:
            assert "common_col" not in mdf.columns
            assert "nonunique_col" not in mdf.columns
        else:
            assert "common_col" in mdf.columns
            if modname != "mod1":
                assert "nonunique_col" in mdf.columns


@pytest.mark.parametrize("drop", (True, False))
def test_pull_oattr_unique(mdata: MuData, oattr: AxisAttr, pull_func_oattr: Callable[..., pd.DataFrame], drop: bool):
    df = pull_func_oattr(common=False, nonunique=False, unique=True, drop=drop)
    assert "mod1:unique_col" in df.columns
    assert len(df.columns) == 8
    assert_dtypes(df, "unique", "mod1:")

    if drop:
        assert "unique_col" not in getattr(mdata["mod1"], oattr)


@pytest.mark.parametrize("drop", (True, False))
def test_pull_oattr_nocommon_nounique(
    mdata: MuData, oattr: AxisAttr, pull_func_oattr: Callable[..., pd.DataFrame], drop: bool
):
    df = pull_func_oattr(common=False, unique=False, drop=drop)
    assert "nonunique_col" not in df.columns
    assert len(df.columns) == (mdata.n_mod - 1) * 5
    assert_dtypes(df, "nonunique", "mod2:")
    assert_dtypes(df, "nonunique", "mod3:")

    for modname, mod in mdata.mod.items():
        mdf = getattr(mod, oattr)
        if drop:
            assert "nonunique_col" not in mdf.columns
        else:
            if modname != "mod1":
                assert "nonunique_col" in mdf.columns


@pytest.mark.parametrize("drop", (True, False))
def test_pull_oattr_nocommon_nounique_join(
    mdata: MuData, oattr: AxisAttr, pull_func_oattr: Callable[..., pd.DataFrame], drop: bool
):
    df = pull_func_oattr(common=False, unique=False, join_nonunique=True, drop=drop)
    assert "nonunique_col" in df.columns
    assert len(df.columns) == 5
    assert_dtypes(df, "nonunique")

    for modname, mod in mdata.mod.items():
        mdf = getattr(mod, oattr)
        if drop:
            assert "nonunique_col" not in mdf.columns
        else:
            if modname != "mod1":
                assert "nonunique_col" in mdf.columns


@pytest.mark.parametrize("drop", (True, False))
def test_pull_oattr_unique_noprefix(
    mdata: MuData, oattr: AxisAttr, pull_func_oattr: Callable[..., pd.DataFrame], drop: bool
):
    df = pull_func_oattr(common=False, nonunique=False, unique=True, prefix_unique=False, drop=drop)
    assert "mod1:unique_col" not in df.columns
    assert "unique_col" in df.columns
    assert len(df.columns) == 8
    assert_dtypes(df, "unique")

    if drop:
        assert "unique_col" not in getattr(mdata["mod1"], oattr).columns
    else:
        assert "unique_col" in getattr(mdata["mod1"], oattr).columns


def test_pull_attr_simple(mdata: MuData, attr: AxisAttr):
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


def test_push_simple(mdata_for_push: MuData, push_func: Callable[..., None], push_df: pd.DataFrame, cattr: AxisAttr):
    push_func()
    for modname, mod in mdata_for_push.mod.items():
        mdf = getattr(mod, cattr)
        assert_dtypes(mdf, "pushed")

        map = getattr(mdata_for_push, f"{cattr}map")[modname].ravel()
        mask = map > 0
        assert (push_df["dtype-int-pushed"][mask] == mdf["dtype-int-pushed"].iloc[map[mask] - 1]).all()

    assert_dtypes(getattr(mdata_for_push["mod2"], cattr), "pushed", "mod2_")
    map = getattr(mdata_for_push, f"{cattr}map")["mod2"].ravel()
    mask = map > 0
    assert (
        push_df["mod2:mod2_dtype-int-pushed"][mask]
        == getattr(mdata_for_push["mod2"], cattr)["mod2_dtype-int-pushed"].iloc[map[mask] - 1]
    ).all()

    push_func(drop=True)
    assert push_df.shape[1] == 0


@pytest.mark.parametrize("drop", (True, False))
def test_push_columns(
    mdata_for_push: MuData, push_func: Callable[..., None], push_df: pd.DataFrame, cattr: AxisAttr, drop: bool
):
    push_func(columns=["dtype-int-pushed", "mod2:mod2_dtype-bool-pushed"], drop=drop)
    for mod in mdata_for_push.mod.values():
        mdf = getattr(mod, cattr)
        assert "dtype-int-pushed" in mdf.columns
        assert "mod2_dtype-bool-pushed" not in mdf.columns
    if drop:
        assert "dtype-int-pushed" not in push_df.columns
        assert "mod2:mod2_dtype-bool-pushed" in push_df.columns

    push_func(columns=["mod2:mod2_dtype-bool-pushed"], mods=["mod2"], drop=drop)
    for modname, mod in mdata_for_push.mod.items():
        mdf = getattr(mod, cattr)
        if modname == "mod2":
            assert "mod2_dtype-bool-pushed" in mdf.columns
        else:
            assert "mod2_dtype-bool-pushed" not in mdf.columns
    if drop:
        assert "mod2:mod2_dtype-bool-pushed" not in mdf.columns

    with pytest.raises(ValueError, match="All mods should be present"):
        push_func(columns=["dtype-int-pushed"], mods=["foo"])


@pytest.mark.parametrize("drop", (True, False))
def test_push_mods(
    mdata_for_push: MuData, push_func: Callable[..., None], push_df: pd.DataFrame, cattr: AxisAttr, drop: bool
):
    push_func(mods="mod2", drop=drop)
    for dtype in ("int", "float", "bool", "string"):
        for modname, mod in mdata_for_push.mod.items():
            mdf = getattr(mod, cattr)
            if modname == "mod2":
                assert f"dtype-{dtype}-pushed" in mdf.columns
                assert f"mod2_dtype-{dtype}-pushed" in mdf.columns
            else:
                assert f"mod2_dtype-{dtype}-pushed" not in mdf.columns
        if drop:
            assert f"dtype-{dtype}-pushed" not in push_df.columns
            assert f"mod2:mod2_dtype-{dtype}-pushed" not in push_df.columns


@pytest.mark.parametrize("drop", (True, False))
def test_push_nocommon(
    mdata_for_push: MuData, push_func: Callable[..., None], push_df: pd.DataFrame, cattr: AxisAttr, drop: bool
):
    push_func(common=False, drop=drop)
    for dtype in ("int", "float", "bool", "string"):
        for modname, mod in mdata_for_push.mod.items():
            mdf = getattr(mod, cattr)
            assert f"dtype-{dtype}-pushed" not in mdf.columns
            if modname == "mod2":
                assert f"mod2_dtype-{dtype}-pushed" in mdf.columns
            else:
                assert f"mod2_dtype-{dtype}-pushed" not in mdf.columns
        if drop:
            assert f"dtype-{dtype}-pushed" in push_df.columns
            assert f"mod2:mod2_dtype-{dtype}-pushed" not in push_df.columns


@pytest.mark.parametrize("drop", (True, False))
def test_push_noprefix(
    mdata_for_push: MuData, push_func: Callable[..., None], push_df: pd.DataFrame, cattr: AxisAttr, drop: bool
):
    push_func(prefixed=False, drop=drop)
    for dtype in ("int", "float", "bool", "string"):
        for mod in mdata_for_push.mod.values():
            mdf = getattr(mod, cattr)
            assert f"dtype-{dtype}-pushed" in mdf.columns
            assert f"mod2_dtype-{dtype}-pushed" not in mdf.columns
        if drop:
            assert f"dtype-{dtype}-pushed" not in push_df.columns


def test_push_drop(mdata_for_push: MuData, push_func: Callable[..., None], push_df: pd.DataFrame, cattr: AxisAttr):
    push_func(only_drop=True)
    assert push_df.shape[1] == 0

    for mod in mdata_for_push.mod.values():
        mdf = getattr(mod, cattr)
        for dtype in ("int", "float", "bool", "string"):
            assert f"dtype-{dtype}-pushed" not in mdf.columns
            assert f"mod2_dtype-{dtype}-pushed" not in mdf.columns
