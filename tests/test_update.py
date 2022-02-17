import unittest
import pytest

import numpy as np
from anndata import AnnData
from mudata import MuData


@pytest.fixture()
def mdata():
    np.random.seed(100)
    mod1 = AnnData(X=np.random.normal(size=1000).reshape(-1, 10))
    mod2 = AnnData(X=np.random.normal(size=1000).reshape(-1, 10))
    batches = np.random.choice(["a", "b", "c"], size=100, replace=True)
    mods = {"mod1": mod1, "mod2": mod2}
    # Make var_names different in different modalities
    for m in ["mod1", "mod2"]:
        mods[m].var_names = [f"{m}_var{i}" for i in range(mods[m].n_vars)]
        mods[m].obs["min_count"] = mods[m].X.min(axis=1)
    mdata = MuData(mods)
    mdata.obs["batch"] = batches
    yield mdata


@pytest.fixture()
def modalities():
    n_mod = 3
    mods = dict()
    np.random.seed(100)
    for i in range(n_mod):
        i1 = i + 1
        m = f"mod{i1}"
        mods[m] = AnnData(X=np.random.normal(size=1000 * i1).reshape(-1, 10 * i1))
        mods[m].obs["mod"] = m
        mods[m].var["mod"] = m
    return mods


@pytest.mark.usefixtures("filepath_h5mu")
class TestMuData:
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

        # Variables are different across modalities
        assert "mod" in mdata.var.columns
        for m, mod in modalities.items():
            # Observations are the same across modalities
            # hence /mod/mod1/obs/mod -> /obs/mod1:mod
            assert f"{m}:mod" in mdata.obs.columns
            # Columns are intact in individual modalities
            assert "mod" in mod.obs.columns
            assert "mod" in mod.var.columns

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

        # Variables are different across modalities
        assert "mod" in mdata.var.columns
        for m, mod in modalities.items():
            # Observations are the same across modalities
            # hence /mod/mod1/obs/mod -> /obs/mod1:mod
            assert f"{m}:mod" in mdata.obs.columns
            # Columns are intact in individual modalities
            assert "mod" in mod.obs.columns
            assert "mod" in mod.var.columns

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

        for m, mod in modalities.items():
            # Observations are the same across modalities
            # hence /mod/mod1/obs/mod -> /obs/mod1:mod
            assert f"{m}:mod" in mdata.obs.columns
            # Variables are intersecting
            # so they won't be merged
            assert f"{m}:mod" in mdata.var.columns
            # Columns are intact in individual modalities
            assert "mod" in mod.obs.columns
            assert "mod" in mod.var.columns

    def test_update_after_filter_obs_adata(self, mdata, filepath_h5mu):
        """
        Check for muon issue #44.
        """
        # Replicate in-place filtering in muon:
        # mu.pp.filter_obs(mdata['mod1'], 'min_count', lambda x: (x < -2))
        mdata.mod["mod1"] = mdata["mod1"][mdata["mod1"].obs["min_count"] < -2].copy()
        mdata.update()
        assert mdata.obs["batch"].isna().sum() == 0


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
