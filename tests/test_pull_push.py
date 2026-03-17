import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from mudata import MuData


@pytest.fixture
def modalities(rng, obs_n, var_unique):
    n_mod = 3
    mods = {}
    for i in range(n_mod):
        i1 = i + 1
        m = f"mod{i1}"
        mods[m] = AnnData(X=rng.normal(size=1000 * i1).reshape(-1, 10 * i1))
        mods[m].obs["mod"] = m
        mods[m].var["mod"] = m

        # common column
        mods[m].var["highly_variable"] = rng.choice([False, True], size=mods[m].n_vars)
        mods[m].obs["common_obs_col"] = rng.integers(0, int(1e6), size=mods[m].n_obs)

        mods[m].obs["dtype-int-common"] = np.arange(mods[m].shape[0])
        mods[m].obs["dtype-float-common"] = np.linspace(0, 1, mods[m].shape[0], dtype=np.float32)
        mods[m].obs["dtype-bool-common"] = rng.choice(1, mods[m].shape[0]).astype(bool)
        mods[m].obs["dtype-categorical-common"] = pd.Categorical(rng.choice(["a", "b", "c"], size=mods[m].shape[0]))
        mods[m].obs["dtype-string-common"] = rng.choice(["a", "b", "c"], size=mods[m].shape[0])

        mods[m].var["dtype-int-common"] = np.arange(mods[m].shape[1])
        mods[m].var["dtype-float-common"] = np.linspace(0, 1, mods[m].shape[1], dtype=np.float32)
        mods[m].var["dtype-bool-common"] = rng.choice(1, mods[m].shape[1]).astype(bool)
        mods[m].var["dtype-categorical-common"] = pd.Categorical(rng.choice(["a", "b", "c"], size=mods[m].shape[1]))
        mods[m].var["dtype-string-common"] = rng.choice(["a", "b", "c"], size=mods[m].shape[1])

        if var_unique:
            mods[m].var_names = [f"mod{m}_var{j}" for j in range(mods[m].n_vars)]

        if i != 0:
            # non-unique column missing from mod1
            mods[m].var["arange"] = np.arange(mods[m].n_vars)
            mods[m].var["dtype-int-nonunique"] = np.arange(mods[m].shape[1])
            mods[m].var["dtype-float-nonunique"] = np.linspace(0, 1, mods[m].shape[1], dtype=np.float32)
            mods[m].var["dtype-bool-nonunique"] = rng.choice(1, mods[m].shape[1]).astype(bool)
            mods[m].var["dtype-categorical-nonunique"] = pd.Categorical(
                rng.choice(["a", "b", "c"], size=mods[m].shape[1])
            )
            mods[m].var["dtype-string-nonunique"] = rng.choice(["a", "b", "c"], size=mods[m].shape[1])
        else:
            # mod1-specific column
            mods[m].var["unique"] = True
            mods[m].var["dtype-int-unique"] = np.arange(mods[m].shape[1])
            mods[m].var["dtype-float-unique"] = np.linspace(0, 1, mods[m].shape[1], dtype=np.float32)
            mods[m].var["dtype-bool-unique"] = rng.choice(1, mods[m].shape[1]).astype(bool)
            mods[m].var["dtype-categorical-unique"] = pd.Categorical(rng.choice(["a", "b", "c"], size=mods[m].shape[1]))
            mods[m].var["dtype-string-unique"] = rng.choice(["a", "b", "c"], size=mods[m].shape[1])

    if obs_n:
        if obs_n == "disjoint":
            mod2_which_obs = rng.choice(mods["mod2"].obs_names, size=mods["mod2"].n_obs // 2, replace=False)
            mods["mod2"] = mods["mod2"][mod2_which_obs].copy()

    return mods


@pytest.fixture
def datasets(rng, var_n, obs_unique):
    n_datasets = 3
    datasets = {}
    for i in range(n_datasets):
        i1 = i + 1
        d = f"dat{i1}"
        datasets[d] = AnnData(X=rng.normal(size=1000 * i1).reshape(10 * i1, -1))
        datasets[d].obs["dataset"] = d
        datasets[d].var["dataset"] = d

        # common column
        datasets[d].obs["common"] = np.tile([False, True], datasets[d].n_obs // 2)

        if i != 0:
            # non-unique column missing from dataset1
            datasets[d].obs["nonunique"] = np.arange(datasets[d].n_obs)
        else:
            # dataset1-specific column
            datasets[d].obs["unique"] = True

        if obs_unique:
            datasets[d].obs_names = [f"dat{d}_obs{j}" for j in range(datasets[d].n_obs)]

    if var_n:
        if var_n == "disjoint":
            dataset2_which_var = rng.choice(datasets["dat2"].var_names, size=datasets["dat2"].n_obs // 2, replace=False)
            datasets["dat2"] = datasets["dat2"][:, dataset2_which_var].copy()

    return datasets


class TestMultiModal:
    @staticmethod
    def assert_dtypes(df, suffix, prefix=""):
        assert pd.api.types.is_integer_dtype(df[f"{prefix}dtype-int-{suffix}"])
        assert pd.api.types.is_float_dtype(df[f"{prefix}dtype-float-{suffix}"])
        assert pd.api.types.is_bool_dtype(df[f"{prefix}dtype-bool-{suffix}"])
        assert pd.api.types.is_categorical_dtype(df[f"{prefix}dtype-categorical-{suffix}"])
        assert (
            pd.api.types.is_string_dtype(df[f"{prefix}dtype-string-{suffix}"])
            or df[f"{prefix}dtype-string-{suffix}"].dtype == object
        )

    @pytest.mark.parametrize("var_unique", [True, False])
    @pytest.mark.parametrize("obs_n", ["joint", "disjoint"])
    def test_pull_var(self, modalities):
        """
        Pulling var annotations shouldn't depend on var_names uniqueness.
        """
        mdata = MuData(modalities)
        mdata.update()
        mdata.pull_var()

        assert "mod" in mdata.var.columns
        for dtype in ("int", "float", "bool", "categorical", "string"):
            assert f"dtype-{dtype}-common" in mdata.var.columns

            assert f"dtype-{dtype}-nonunique" not in mdata.var.columns
            assert f"dtype-{dtype}-unique" not in mdata.var.columns
        self.assert_dtypes(mdata.var, "common")

        for m, mod in modalities.items():
            # Annotations are correct
            assert all(mdata.var.loc[mdata.varmap[m] > 0, "mod"] == m)
            # Columns are intact in individual modalities
            assert "mod" in mod.var.columns
            assert all(mod.var["mod"] == m)

        # Clean .var
        mdata.var = mdata.var.loc[:, []]

        # Pull a common column
        mdata.pull_var(columns=["highly_variable"])
        assert "highly_variable" in mdata.var.columns
        assert (~pd.isnull(mdata.var.highly_variable)).sum() == mdata.n_vars
        mdata.var = mdata.var.loc[:, []]

        # Pull a common column from one modality
        mdata.pull_var(columns=["highly_variable"], mods=["mod2"])
        assert "mod2:highly_variable" in mdata.var.columns
        assert (~pd.isnull(mdata.var["mod2:highly_variable"])).sum() == mdata["mod2"].n_vars
        mdata.var = mdata.var.loc[:, []]

        # do not pull unique columns
        mdata.pull_var(common=True, nonunique=True, unique=False)
        assert "mod1:unique" not in mdata.var.columns
        assert "highly_variable" in mdata.var.columns
        assert "arange" not in mdata.var.columns

        self.assert_dtypes(mdata.var, "common")
        self.assert_dtypes(mdata.var, "nonunique", "mod2:")
        self.assert_dtypes(mdata.var, "nonunique", "mod3:")
        mdata.var = mdata.var.loc[:, []]

        # only pull a unique column
        mdata.pull_var(common=False, nonunique=False, unique=True)
        assert "mod1:unique" in mdata.var.columns
        assert len(mdata.var.columns) == 6
        self.assert_dtypes(mdata.var, "unique", "mod1:")
        mdata.var = mdata.var.loc[:, []]

        # pull non-unique but do not join
        mdata.pull_var(common=False, unique=False)
        assert "arange" not in mdata.var.columns
        assert len(mdata.var.columns) == (mdata.n_mod - 1) * 6
        self.assert_dtypes(mdata.var, "nonunique", "mod2:")
        self.assert_dtypes(mdata.var, "nonunique", "mod3:")
        mdata.var = mdata.var.loc[:, []]

        # pull non-unique and join them
        mdata.pull_var(common=False, unique=False, join_nonunique=True)
        assert "arange" in mdata.var.columns
        assert len(mdata.var.columns) == 6
        self.assert_dtypes(mdata.var, "nonunique")
        mdata.var = mdata.var.loc[:, []]

        # pull unique and do not prefix
        mdata.pull_var(common=False, nonunique=False, unique=True, prefix_unique=False)
        assert "mod1:unique" not in mdata.var.columns
        assert "unique" in mdata.var.columns
        self.assert_dtypes(mdata.var, "unique")
        assert len(mdata.var.columns) == 6
        mdata.var = mdata.var.loc[:, []]

    @pytest.mark.parametrize("var_unique", [True, False])
    @pytest.mark.parametrize("obs_n", ["joint", "disjoint"])
    def test_pull_obs_simple(self, modalities):
        """
        Pulling obs annotations.
        """
        mdata = MuData(modalities)
        mdata.update()

        mdata.pull_obs()

        # pulling should work
        for m in mdata.mod.keys():
            assert f"{m}:mod" in mdata.obs.columns
            assert f"{m}:common_obs_col" in mdata.obs.columns

            for dtype in ("int", "float", "bool", "categorical", "string"):
                assert f"{m}:dtype-{dtype}-common" in mdata.obs.columns

            self.assert_dtypes(mdata.obs, "common", f"{m}:")

            modmap = mdata.obsmap[m].ravel()
            mask = modmap > 0
            assert (
                mdata.obs[f"{m}:common_obs_col"][mask].to_numpy()
                == mdata.mod[m].obs["common_obs_col"].to_numpy()[modmap[mask] - 1]
            ).all()

        # join_common shouldn't work
        with pytest.raises(ValueError, match="shared obs_names"):
            mdata.pull_obs(join_common=True)

        # join_nonunique shouldn't work
        with pytest.raises(ValueError, match="shared obs_names"):
            mdata.pull_obs(join_nonunique=True)

    @pytest.mark.parametrize("var_unique", [True, False])
    @pytest.mark.parametrize("obs_n", ["joint", "disjoint"])
    def test_push_var_simple(self, rng, modalities):
        """
        Pushing var annotations.
        """
        mdata = MuData(modalities)
        mdata.update()

        mdata.var["dtype-int-pushed"] = np.arange(mdata.shape[1])
        mdata.var["dtype-float-pushed"] = np.linspace(0, 1, mdata.shape[1], dtype=np.float32)
        mdata.var["dtype-bool-pushed"] = rng.choice(1, mdata.shape[1]).astype(bool)
        mdata.var["dtype-categorical-pushed"] = pd.Categorical(rng.choice(["a", "b", "c"], size=mdata.shape[1]))
        mdata.var["dtype-string-pushed"] = rng.choice(["a", "b", "c"], size=mdata.shape[1])
        mdata.var["mod2:mod2_dtype-int-pushed"] = np.arange(mdata.shape[1])
        mdata.var["mod2:mod2_dtype-float-pushed"] = np.linspace(0, 1, mdata.shape[1], dtype=np.float32)
        mdata.var["mod2:mod2_dtype-bool-pushed"] = rng.choice(1, mdata.shape[1]).astype(bool)
        mdata.var["mod2:mod2_dtype-categorical-pushed"] = pd.Categorical(
            rng.choice(["a", "b", "c"], size=mdata.shape[1])
        )
        mdata.var["mod2:mod2_dtype-string-pushed"] = rng.choice(["a", "b", "c"], size=mdata.shape[1])
        mdata.push_var()

        # pushing should work
        for modname, mod in mdata.mod.items():
            self.assert_dtypes(mod.var, "pushed")

            map = mdata.varmap[modname].ravel()
            mask = map > 0
            assert (mdata.var["dtype-int-pushed"][mask] == mod.var["dtype-int-pushed"].iloc[map[mask] - 1]).all()

        self.assert_dtypes(mdata["mod2"].var, "pushed", "mod2_")
        map = mdata.varmap["mod2"].ravel()
        mask = map > 0
        assert (
            mdata.var["mod2:mod2_dtype-int-pushed"][mask]
            == mdata["mod2"].var["mod2_dtype-int-pushed"].iloc[map[mask] - 1]
        ).all()

    @pytest.mark.parametrize("var_unique", [True, False])
    @pytest.mark.parametrize("obs_n", ["joint", "disjoint"])
    def test_push_obs_simple(self, rng, modalities):
        """
        Pushing obs annotations.
        """
        mdata = MuData(modalities)
        mdata.update()

        mdata.obs["pushed"] = rng.integers(0, int(1e6), size=mdata.n_obs)
        mdata.obs["mod2:mod2_pushed"] = rng.integers(0, int(1e6), size=mdata.n_obs)
        mdata.push_obs()

        # pushing should work
        for modname, mod in mdata.mod.items():
            assert "pushed" in mod.obs.columns

            map = mdata.obsmap[modname].ravel()
            mask = map > 0
            assert (mdata.obs["pushed"][mask] == mod.obs["pushed"].iloc[map[mask] - 1]).all()

        assert "mod2_pushed" in mdata["mod2"].obs.columns
        map = mdata.obsmap["mod2"].ravel()
        mask = map > 0
        assert (mdata.obs["mod2:mod2_pushed"][mask] == mdata["mod2"].obs["mod2_pushed"].iloc[map[mask] - 1]).all()


class TestMultiDataset:
    @pytest.mark.parametrize("obs_unique", [True, False])
    @pytest.mark.parametrize("var_n", ["joint", "disjoint"])
    def test_pull_obs(self, datasets):
        """
        Pulling obs annotations shouldn't depend on obs_names uniqueness.
        """
        mdata = MuData(datasets, axis=1)
        mdata.update()

        mdata.pull_obs()

        assert "dataset" in mdata.obs.columns
        assert "common" in mdata.obs.columns
        assert "nonunique" not in mdata.obs.columns
        assert "unique" not in mdata.obs.columns

        for m, mod in datasets.items():
            # Annotations are correct
            assert all(mdata.obs.loc[mdata.obsmap[m] > 0, "dataset"] == m)
            # Columns are intact in individual modalities
            assert "dataset" in mod.obs.columns
            assert all(mod.obs["dataset"] == m)

        # Clean .obs
        mdata.obs = mdata.obs.loc[:, []]

        # Pull a common column
        mdata.pull_obs(columns=["common"])
        assert "common" in mdata.obs.columns
        assert (~pd.isnull(mdata.obs.common)).sum() == mdata.n_obs
        mdata.obs = mdata.obs.loc[:, []]

        # Pull a common column from one modality
        mdata.pull_obs(columns=["common"], mods=["dat2"])
        assert "dat2:common" in mdata.obs.columns
        assert (~pd.isnull(mdata.obs["dat2:common"])).sum() == mdata["dat2"].n_obs
        mdata.obs = mdata.obs.loc[:, []]

        # do not pull unique columns
        mdata.pull_obs(common=True, nonunique=True, unique=False)
        assert "dat1:unique" not in mdata.obs.columns
        assert "common" in mdata.obs.columns
        assert "nonunique" not in mdata.obs.columns
        mdata.obs = mdata.obs.loc[:, []]

        # only pull a unique column
        mdata.pull_obs(common=False, nonunique=False, unique=True)
        assert "dat1:unique" in mdata.obs.columns
        assert len(mdata.obs.columns) == 1
        mdata.obs = mdata.obs.loc[:, []]

        # pull non-unique but do not join
        mdata.pull_obs(common=False, unique=False)
        assert "nonunique" not in mdata.obs.columns
        assert len(mdata.obs.columns) == mdata.n_mod - 1
        mdata.obs = mdata.obs.loc[:, []]

        # pull non-unique and join them
        mdata.pull_obs(common=False, unique=False, join_nonunique=True)
        assert "nonunique" in mdata.obs.columns
        assert len(mdata.obs.columns) == 1
        mdata.obs = mdata.obs.loc[:, []]

        # pull unique and do not prefix
        mdata.pull_obs(common=False, nonunique=False, unique=True, prefix_unique=False)
        assert "dat1:unique" not in mdata.obs.columns
        assert "unique" in mdata.obs.columns
        assert len(mdata.obs.columns) == 1
        mdata.obs = mdata.obs.loc[:, []]

    @pytest.mark.parametrize("obs_unique", [True, False])
    @pytest.mark.parametrize("var_n", ["joint", "disjoint"])
    def test_pull_var_simple(self, datasets):
        """
        Pulling var annotations.
        """
        mdata = MuData(datasets, axis=1)
        mdata.update()

        mdata.pull_var()

        # pulling should work
        for m in mdata.mod.keys():
            assert f"{m}:dataset" in mdata.var.columns

        # join_common shouldn't work
        with pytest.raises(ValueError, match="shared var_names"):
            mdata.pull_var(join_common=True)

        # join_nonunique shouldn't work
        with pytest.raises(ValueError, match="shared var_names"):
            mdata.pull_var(join_nonunique=True)
