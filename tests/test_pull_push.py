import unittest

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from mudata import MuData


@pytest.fixture()
def modalities(request, obs_n, var_unique):
    n_mod = 3
    mods = dict()
    np.random.seed(100)
    for i in range(n_mod):
        i1 = i + 1
        m = f"mod{i1}"
        mods[m] = AnnData(X=np.random.normal(size=1000 * i1).reshape(-1, 10 * i1))
        mods[m].obs["mod"] = m
        mods[m].var["mod"] = m

        # common column
        mods[m].var["highly_variable"] = np.tile([False, True], mods[m].n_vars // 2)

        if var_unique:
            mods[m].var_names = [f"mod{m}_var{j}" for j in range(mods[m].n_vars)]

        if i != 0:
            # non-unique column missing from mod1
            mods[m].var["arange"] = np.arange(mods[m].n_vars)
        else:
            # mod1-specific column
            mods[m].var["unique"] = True

    if obs_n:
        if obs_n == "disjoint":
            mod2_which_obs = np.random.choice(
                mods["mod2"].obs_names, size=mods["mod2"].n_obs // 2, replace=False
            )
            mods["mod2"] = mods["mod2"][mod2_which_obs].copy()

    return mods


@pytest.fixture()
def datasets(request, var_n, obs_unique):
    n_datasets = 3
    datasets = dict()
    np.random.seed(100)
    for i in range(n_datasets):
        i1 = i + 1
        d = f"dat{i1}"
        datasets[d] = AnnData(X=np.random.normal(size=1000 * i1).reshape(10 * i1, -1))
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
            dataset2_which_var = np.random.choice(
                datasets["dat2"].var_names, size=datasets["dat2"].n_obs // 2, replace=False
            )
            datasets["dat2"] = datasets["dat2"][:, dataset2_which_var].copy()

    return datasets


@pytest.mark.usefixtures("filepath_h5mu")
class TestMultiModal:
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
        assert "highly_variable" in mdata.var.columns
        assert "arange" not in mdata.var.columns
        assert "unique" not in mdata.var.columns

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
        mdata.var = mdata.var.loc[:, []]

        # only pull a unique column
        mdata.pull_var(common=False, nonunique=False, unique=True)
        assert "mod1:unique" in mdata.var.columns
        assert len(mdata.var.columns) == 1
        mdata.var = mdata.var.loc[:, []]

        # pull non-unique but do not join
        mdata.pull_var(common=False, unique=False)
        assert "arange" not in mdata.var.columns
        assert len(mdata.var.columns) == mdata.n_mod - 1
        mdata.var = mdata.var.loc[:, []]

        # pull non-unique and join them
        mdata.pull_var(common=False, unique=False, join_nonunique=True)
        assert "arange" in mdata.var.columns
        assert len(mdata.var.columns) == 1
        mdata.var = mdata.var.loc[:, []]

        # pull unique and do not prefix
        mdata.pull_var(common=False, nonunique=False, unique=True, prefix_unique=False)
        assert "mod1:unique" not in mdata.var.columns
        assert "unique" in mdata.var.columns
        assert len(mdata.var.columns) == 1
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

        # join_common shouldn't work
        with pytest.raises(ValueError, match="shared obs_names"):
            mdata.pull_obs(join_common=True)

        # join_nonunique shouldn't work
        with pytest.raises(ValueError, match="shared obs_names"):
            mdata.pull_obs(join_nonunique=True)

    @pytest.mark.parametrize("var_unique", [True, False])
    @pytest.mark.parametrize("obs_n", ["joint", "disjoint"])
    def test_push_var_simple(self, modalities):
        """
        Pushing var annotations.
        """
        mdata = MuData(modalities)
        mdata.update()

        mdata.var["pushed"] = True
        mdata.var["mod2:mod2_pushed"] = True
        mdata.push_var()

        # pushing should work
        for mod in mdata.mod.values():
            assert "pushed" in mod.var.columns
        assert "mod2_pushed" in mdata["mod2"].var.columns

    @pytest.mark.parametrize("var_unique", [True, False])
    @pytest.mark.parametrize("obs_n", ["joint", "disjoint"])
    def test_push_obs_simple(self, modalities):
        """
        Pushing obs annotations.
        """
        mdata = MuData(modalities)
        mdata.update()

        mdata.obs["pushed"] = True
        mdata.obs["mod2:mod2_pushed"] = True
        mdata.push_obs()

        # pushing should work
        for mod in mdata.mod.values():
            assert "pushed" in mod.obs.columns
        assert "mod2_pushed" in mdata["mod2"].obs.columns


@pytest.mark.usefixtures("filepath_h5mu")
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


if __name__ == "__main__":
    unittest.main()
