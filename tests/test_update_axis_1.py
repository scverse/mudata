import unittest

import numpy as np
import pytest
from anndata import AnnData

from mudata import MuData


@pytest.fixture()
def mdata(request, var_n, var_across, var_mod):
    # Generate unique, intersecting, and joint observations by default
    np.random.seed(100)
    ad1 = AnnData(X=np.random.normal(size=1000).reshape(-1, 10))
    ad2 = AnnData(X=np.random.normal(size=1000).reshape(-1, 10))

    datasets = {"set1": ad1, "set2": ad2}
    # Make obs_names different in different datasets
    for dname, d in datasets.items():
        datasets[dname].obs_names = [f"{d}_obs{i}" for i in range(d.n_obs)]
        datasets[dname].var_names = [f"var{i}" for i in range(d.n_vars)]
        datasets[dname].obs["min_count"] = d.X.min(axis=1)

    if var_n:
        if var_n == "disjoint":
            set2_which_var = np.random.choice(
                datasets["set2"].var_names, size=datasets["set2"].n_vars // 2, replace=False
            )
            datasets["set2"] = datasets["set2"][:, set2_which_var].copy()

    if var_across:
        if var_across != "intersecting":
            raise NotImplementedError("Tests for non-intersecting var_names are not implemented")

    if var_mod:
        if var_mod == "duplicated":
            for dname, d in datasets.itmes():
                # Index does not support mutable operations
                var_names = d.var_names.values.copy()
                var_names[1] = var_names[0]
                datasets[dname].var_names = var_names

    mdata = MuData(datasets, axis=1)

    genesets = np.random.choice(["a", "b", "c"], size=mdata.n_vars, replace=True)
    mdata.var["geneset"] = genesets

    return mdata


@pytest.fixture()
def datasets(request, var_n, var_across, var_mod):
    n_sets = 3
    datasets = dict()
    np.random.seed(100)
    for i in range(n_sets):
        i1 = i + 1
        d = f"set{i1}"
        datasets[d] = AnnData(X=np.random.normal(size=1000 * i1).reshape(-1, 10 * i1))
        datasets[d].obs["dataset"] = d
        datasets[d].var["dataset"] = d

    if var_n:
        if var_n == "disjoint":
            set2_which_var = np.random.choice(
                datasets["set2"].var_names, size=datasets["set2"].n_vars // 2, replace=False
            )
            datasets["set2"] = datasets["set2"][set2_which_var].copy()

    if var_across:
        if var_across != "intersecting":
            raise NotImplementedError("Tests for non-intersecting var_names are not implemented")

    if var_mod:
        if var_mod == "duplicated":
            for dname, d in datasets.itmes():
                # Index does not support mutable operations
                var_names = d.var_names.values.copy()
                var_names[1] = var_names[0]
                datasets[dname].var_names = var_names

    return datasets


@pytest.mark.usefixtures("filepath_h5mu")
class TestMuData:
    @pytest.mark.parametrize("var_mod", ["unique"])
    @pytest.mark.parametrize("var_across", ["intersecting"])
    @pytest.mark.parametrize("var_n", ["joint", "disjoint"])
    def test_update_simple(self, datasets):
        """
        Update should work when
        - var_names are the same across datasets,
        - obs_names are unique to each dataset
        """
        for d, dset in datasets.items():
            datasets[d].obs_names = [f"{d}_obs{j}" for j in range(dset.n_obs)]
        mdata = MuData(datasets, axis=1)
        mdata.update()

        # Variables are different across datasets
        assert "dataset" in mdata.obs.columns
        for d, dset in datasets.items():
            # Veriables are the same across datasets
            # hence /mod/mod1/var/dataset -> /var/mod1:dataset
            assert f"{d}:dataset" in mdata.var.columns
            # Columns are intact in individual datasets
            assert "dataset" in dset.obs.columns
            assert all(dset.obs["dataset"] == d)
            assert "dataset" in dset.var.columns
            assert all(dset.var["dataset"] == d)

    @pytest.mark.parametrize("var_mod", ["unique"])
    @pytest.mark.parametrize("var_across", ["intersecting"])
    @pytest.mark.parametrize("var_n", ["joint", "disjoint"])
    def test_update_duplicates(self, datasets):
        """
        Update should work when
        - var_names are the same across modalities,
        - there are duplicated obs_names, which are not intersecting
          between modalities
        """
        for d, dset in datasets.items():
            dset.obs_names = [f"{d}_obs{j // 2}" for j in range(dset.n_obs)]
        mdata = MuData(datasets, axis=1)
        mdata.update()

        # Observations are different across datasets
        assert "dataset" in mdata.obs.columns
        for d, dset in datasets.items():
            # Variables are the same across datasets
            # hence /mod/mod1/var/datasets -> /var/mod1:datasets
            assert f"{d}:dataset" in mdata.var.columns
            # Columns are intact in individual modalities
            assert "dataset" in dset.obs.columns
            assert all(dset.obs["dataset"] == d)
            assert "dataset" in dset.var.columns
            assert all(dset.var["dataset"] == d)

    @pytest.mark.parametrize("var_mod", ["unique"])
    @pytest.mark.parametrize("var_across", ["intersecting"])
    @pytest.mark.parametrize("var_n", ["joint", "disjoint"])
    def test_update_intersecting(self, datasets):
        """
        Update should work when
        - var_names are the same across datasets,
        - there are intersecting obs_names
        """
        for d, dset in datasets.items():
            # [mod1] var0, mod1_var1, mod1_var2, ...; [mod2] var0, mod2_var1, mod2_var2, ...
            dset.obs_names = [f"{d}_obs{j}" if j != 0 else f"obs_{j}" for j in range(dset.n_obs)]
        mdata = MuData(datasets, axis=1)
        mdata.update()

        for d, dset in datasets.items():
            # Veriables are the same across datasets
            # hence /mod/mod1/var/dataset -> /var/mod1:dataset
            assert f"{d}:dataset" in mdata.var.columns
            # Observations are intersecting
            # so they won't be merged
            assert f"{d}:dataset" in mdata.obs.columns
            # Columns are intact in individual modalities
            assert "dataset" in dset.obs.columns
            assert all(dset.obs["dataset"] == d)
            assert "dataset" in dset.var.columns
            assert all(dset.var["dataset"] == d)

    @pytest.mark.parametrize("var_mod", ["unique"])
    @pytest.mark.parametrize("var_across", ["intersecting"])
    @pytest.mark.parametrize("var_n", ["joint", "disjoint"])
    def test_update_after_var_reordered(self, mdata):
        """
        Update should work if var are reordered.
        """
        mdata.varm["test_varm"] = np.random.normal(size=(mdata.n_vars, 2))

        some_var_names = mdata.var_names.values[:2]

        true_varm_values = [
            mdata.varm["test_varm"][np.where(mdata.var_names.values == name)[0][0]]
            for name in some_var_names
        ]

        mdata.mod["set1"] = mdata["set1"][:, ::-1].copy()
        mdata.update()

        test_varm_values = [
            mdata.varm["test_varm"][np.where(mdata.var_names == name)[0][0]]
            for name in some_var_names
        ]

        assert all(
            [all(true_varm_values[i] == test_varm_values[i]) for i in range(len(true_varm_values))]
        )


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
