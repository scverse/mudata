import unittest

import numpy as np
import pytest
from anndata import AnnData

from mudata import MuData


@pytest.fixture()
def mdata(request, obs_n, obs_across, obs_mod):
    # Generate unique, intersecting, and joint observations by default
    np.random.seed(100)
    mod1 = AnnData(X=np.random.normal(size=1000).reshape(-1, 10))
    mod2 = AnnData(X=np.random.normal(size=1000).reshape(-1, 10))

    mods = {"mod1": mod1, "mod2": mod2}
    # Make var_names different in different modalities
    for m in ["mod1", "mod2"]:
        mods[m].obs_names = [f"obs{i}" for i in range(mods[m].n_obs)]
        mods[m].var_names = [f"{m}_var{i}" for i in range(mods[m].n_vars)]
        mods[m].obs["min_count"] = mods[m].X.min(axis=1)

    if obs_n:
        if obs_n == "disjoint":
            mod2_which_obs = np.random.choice(
                mods["mod2"].obs_names, size=mods["mod2"].n_obs // 2, replace=False
            )
            mods["mod2"] = mods["mod2"][mod2_which_obs].copy()

    if obs_across:
        if obs_across != "intersecting":
            raise NotImplementedError("Tests for non-intersecting obs_names are not implemented")

    if obs_mod:
        if obs_mod == "duplicated":
            for m in ["mod1", "mod2"]:
                # Index does not support mutable operations
                obs_names = mods[m].obs_names.values.copy()
                obs_names[1] = obs_names[0]
                mods[m].obs_names = obs_names

    mdata = MuData(mods)

    batches = np.random.choice(["a", "b", "c"], size=mdata.shape[0], replace=True)
    mdata.obs["batch"] = batches

    return mdata


@pytest.fixture()
def modalities(request, obs_n, obs_across, obs_mod):
    n_mod = 3
    mods = dict()
    np.random.seed(100)
    for i in range(n_mod):
        i1 = i + 1
        m = f"mod{i1}"
        mods[m] = AnnData(X=np.random.normal(size=1000 * i1).reshape(-1, 10 * i1))
        mods[m].obs["mod"] = m
        mods[m].var["mod"] = m

    if obs_n:
        if obs_n == "disjoint":
            mod2_which_obs = np.random.choice(
                mods["mod2"].obs_names, size=mods["mod2"].n_obs // 2, replace=False
            )
            mods["mod2"] = mods["mod2"][mod2_which_obs].copy()

    if obs_across:
        if obs_across != "intersecting":
            raise NotImplementedError("Tests for non-intersecting obs_names are not implemented")

    if obs_mod:
        if obs_mod == "duplicated":
            for m in ["mod1", "mod2"]:
                # Index does not support mutable operations
                obs_names = mods[m].obs_names.values.copy()
                obs_names[1] = obs_names[0]
                mods[m].obs_names = obs_names

    return mods


@pytest.mark.usefixtures("filepath_h5mu")
class TestMuData:
    @pytest.mark.parametrize("obs_mod", ["unique"])
    @pytest.mark.parametrize("obs_across", ["intersecting"])
    @pytest.mark.parametrize("obs_n", ["joint", "disjoint"])
    def test_update_simple(self, modalities):
        """
        Update should work when
        - obs_names are the same across modalities,
        - var_names are unique to each modality
        """
        for m, mod in modalities.items():
            mod.var_names = [f"{m}_var{j}" for j in range(mod.n_vars)]
        mdata = MuData(modalities)
        mdata.update()
        mdata.pull_obs()
        mdata.pull_var()

        # Variables are different across modalities
        assert "mod" in mdata.var.columns
        for m, mod in modalities.items():
            # Observations are the same across modalities
            # hence /mod/mod1/obs/mod -> /obs/mod1:mod
            assert f"{m}:mod" in mdata.obs.columns
            # Columns are intact in individual modalities
            assert "mod" in mod.obs.columns
            assert all(mod.obs["mod"] == m)
            assert "mod" in mod.var.columns
            assert all(mod.var["mod"] == m)

    @pytest.mark.parametrize("obs_mod", ["unique"])
    @pytest.mark.parametrize("obs_across", ["intersecting"])
    @pytest.mark.parametrize("obs_n", ["joint", "disjoint"])
    def test_update_duplicates(self, modalities):
        """
        Update should work when
        - obs_names are the same across modalities,
        - there are duplicated var_names, which are not intersecting
          between modalities
        """
        for m, mod in modalities.items():
            mod.var_names = [f"{m}_var{j // 2}" for j in range(mod.n_vars)]
        mdata = MuData(modalities)
        mdata.update()
        mdata.pull_obs()
        mdata.pull_var()

        # Variables are different across modalities
        assert "mod" in mdata.var.columns
        for m, mod in modalities.items():
            # Observations are the same across modalities
            # hence /mod/mod1/obs/mod -> /obs/mod1:mod
            assert f"{m}:mod" in mdata.obs.columns
            # Columns are intact in individual modalities
            assert "mod" in mod.obs.columns
            assert all(mod.obs["mod"] == m)
            assert "mod" in mod.var.columns
            assert all(mod.var["mod"] == m)

    @pytest.mark.parametrize("obs_mod", ["unique"])
    @pytest.mark.parametrize("obs_across", ["intersecting"])
    @pytest.mark.parametrize("obs_n", ["joint", "disjoint"])
    def test_update_intersecting(self, modalities):
        """
        Update should work when
        - obs_names are the same across modalities,
        - there are intersecting var_names,
          which are unique in each modality
        """
        for m, mod in modalities.items():
            # [mod1] var0, mod1_var1, mod1_var2, ...; [mod2] var0, mod2_var1, mod2_var2, ...
            mod.var_names = [f"{m}_var{j}" if j != 0 else f"var_{j}" for j in range(mod.n_vars)]
        mdata = MuData(modalities)
        mdata.update()
        mdata.pull_obs()
        # New behaviour since v0.4:
        # - Will add a single column 'mod' with the correct labels even with intersecting var_names
        mdata.pull_var()
        # - Will add the columns with modality prefixes
        mdata.pull_var(join_common=False)

        for m, mod in modalities.items():
            # Observations are the same across modalities
            # hence /mod/mod1/obs/mod -> /obs/mod1:mod
            assert f"{m}:mod" in mdata.obs.columns
            # Variables are intersecting
            # so they won't be merged
            assert f"{m}:mod" in mdata.var.columns
            # Columns are intact in individual modalities
            assert "mod" in mod.obs.columns
            assert all(mod.obs["mod"] == m)
            assert "mod" in mod.var.columns
            assert all(mod.var["mod"] == m)

    @pytest.mark.parametrize("obs_mod", ["unique"])
    @pytest.mark.parametrize("obs_across", ["intersecting"])
    @pytest.mark.parametrize("obs_n", ["joint", "disjoint"])
    def test_update_after_filter_obs_adata(self, mdata):
        """
        Check for muon issue #44.
        """
        # Replicate in-place filtering in muon:
        # mu.pp.filter_obs(mdata['mod1'], 'min_count', lambda x: (x < -2))
        mdata.mod["mod1"] = mdata["mod1"][mdata["mod1"].obs["min_count"] < -2].copy()
        mdata.update()
        mdata.pull_obs()
        assert mdata.obs["batch"].isna().sum() == 0

    @pytest.mark.parametrize("obs_mod", ["unique"])
    @pytest.mark.parametrize("obs_across", ["intersecting"])
    @pytest.mark.parametrize("obs_n", ["joint", "disjoint"])
    def test_update_after_obs_reordered(self, mdata):
        """
        Update should work if obs are reordered.
        """
        mdata.obsm["test_obsm"] = np.random.normal(size=(mdata.n_obs, 2))

        some_obs_names = mdata.obs_names.values[:2]

        true_obsm_values = [
            mdata.obsm["test_obsm"][np.where(mdata.obs_names.values == name)[0][0]]
            for name in some_obs_names
        ]

        mdata.mod["mod1"] = mdata["mod1"][::-1].copy()
        mdata.update()

        test_obsm_values = [
            mdata.obsm["test_obsm"][np.where(mdata.obs_names == name)[0][0]]
            for name in some_obs_names
        ]

        assert all(
            [all(true_obsm_values[i] == test_obsm_values[i]) for i in range(len(true_obsm_values))]
        )

    @pytest.mark.parametrize("obs_mod", ["unique"])
    @pytest.mark.parametrize("obs_across", ["intersecting"])
    @pytest.mark.parametrize("obs_n", ["joint", "disjoint"])
    def test_update_intersecting_var_names_after_filtering(self, mdata):
        orig_shape = mdata.shape
        mdata.mod["mod1"].var_names = [str(i) for i in range(mdata["mod1"].n_vars)]
        mdata.mod["mod2"].var_names = [str(i) for i in range(mdata["mod2"].n_vars)]
        mdata.update()
        mdata.mod["mod1"] = mdata["mod1"][:, :5].copy()
        mdata["mod1"].var["true"] = True
        mdata["mod2"].var["false"] = False
        assert mdata["mod1"].n_vars == 5
        mdata.update()
        mdata.pull_var(prefix_unique=False)
        assert mdata.n_obs == orig_shape[0]
        assert mdata.n_vars == mdata["mod1"].n_vars + mdata["mod2"].n_vars
        assert mdata.var["true"].sum() == 5
        assert (~mdata.var["false"]).sum() == (~mdata["mod2"].var["false"]).sum()

    @pytest.mark.parametrize("obs_mod", ["unique"])
    @pytest.mark.parametrize("obs_across", ["intersecting"])
    @pytest.mark.parametrize("obs_n", ["joint", "disjoint"])
    def test_update_to_new_names(self, mdata):
        mdata["mod1"].var_names = [f"_mod1_var{i}" for i in range(1, mdata["mod1"].n_vars + 1)]
        mdata["mod2"].var_names = [f"_mod2_var{i}" for i in range(1, mdata["mod2"].n_vars + 1)]
        mdata.update()


# @pytest.mark.usefixtures("filepath_h5mu")
# class TestMuDataSameVars:
#     def test_update_simple(self, modalities):
#         """
#         Update should work when
#         - obs_names are the same across modalities,
#         - var_names are unique to each modality
#         """
#         for m, mod in modalities.items():
#             mod.var_names = [f"{m}var_{j}" for j in range(mod.n_vars)]
#         mdata = MuData(modalities, axis=0)
#         mdata.update()

#         # Observations are the same across modalities
#         # hence /mod/mod1/obs/mod -> /obs/mod1:mod
#         assert f"{m}:mod" in mdata.obs.columns
#         # Variables are different across modalities
#         assert "mod" in mdata.var.columns
#         for m, mod in modalities.items():
#             # Columns are intact in individual modalities
#             assert "mod" in mod.obs.columns
#             assert "mod" in mod.var.columns


if __name__ == "__main__":
    unittest.main()
