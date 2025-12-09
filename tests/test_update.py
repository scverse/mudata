import unittest
from functools import reduce

import numpy as np
import pytest
from anndata import AnnData

from mudata import MuData, set_options


@pytest.fixture()
def modalities(request, obs_n, obs_across, obs_mod):
    n_mod = 3
    mods = dict()
    np.random.seed(100)
    for i in range(n_mod):
        i1 = i + 1
        m = f"mod{i1}"
        mods[m] = AnnData(X=np.random.normal(size=3000 * i1).reshape(-1, 10 * i1))
        mods[m].obs["mod"] = m
        mods[m].var["mod"] = m
        mods[m].obs["min_count"] = mods[m].X.min(axis=1)

    if obs_n:
        if obs_n == "disjoint":
            mod2_which_obs = np.random.choice(
                mods["mod1"].obs_names, size=mods["mod1"].n_obs // 2, replace=False
            )
            mods["mod1"] = mods["mod1"][mod2_which_obs].copy()

    if obs_across:
        if obs_across != "intersecting":
            raise NotImplementedError("Tests for non-intersecting obs_names are not implemented")

    if obs_mod:
        if obs_mod == "duplicated":
            obsnames2 = mods["mod2"].obs_names.to_numpy()
            obsnames3 = mods["mod3"].obs_names.to_numpy()
            varnames2 = mods["mod2"].var_names.to_numpy()
            varnames3 = mods["mod3"].var_names.to_numpy()
            obsnames2[0] = obsnames2[1] = obsnames3[1] = "testobs"
            varnames2[0] = varnames2[1] = varnames3[1] = "testvar"
            mods["mod2"].obs_names = obsnames2
            mods["mod3"].obs_names = obsnames3
            mods["mod2"].var_names = varnames2
            mods["mod3"].var_names = varnames3
        elif (
            obs_mod == "extreme_duplicated"
        ):  # integer overflow: https://github.com/scverse/mudata/issues/107
            obsnames2 = mods["mod2"].obs_names.to_numpy()
            varnames2 = mods["mod2"].var_names.to_numpy()
            obsnames2[:-1] = obsnames2[0] = "testobs"
            varnames2[:-1] = varnames2[0] = "testvar"
            mods["mod2"].obs_names = obsnames2
            mods["mod2"].var_names = varnames2

    return mods


@pytest.fixture()
def mdata_legacy(modalities):
    mdata = MuData(modalities)

    batches = np.random.choice(["a", "b", "c"], size=mdata.shape[0], replace=True)
    mdata.obs["batch"] = batches

    return mdata


@pytest.fixture()
def mdata(modalities, axis):
    md = MuData(modalities, axis=axis)

    md.obs["batch"] = np.random.choice(["a", "b", "c"], size=md.shape[0], replace=True)
    md.var["batch"] = np.random.choice(["d", "e", "f"], size=md.shape[1], replace=True)

    md.obsm["test"] = np.random.normal(size=(md.n_obs, 2))
    md.varm["test"] = np.random.normal(size=(md.n_var, 2))

    return md


@pytest.mark.usefixtures("filepath_h5mu")
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("obs_mod", ["unique", "duplicated", "extreme_duplicated"])
@pytest.mark.parametrize("obs_across", ["intersecting"])
@pytest.mark.parametrize("obs_n", ["joint", "disjoint"])
class TestMuData:
    @pytest.fixture(autouse=True)
    def new_update(self):
        set_options(pull_on_update=False)
        yield
        set_options(pull_on_update=None)

    @staticmethod
    def get_attrm_values(mdata, attr, key, names):
        attrm = getattr(mdata, f"{attr}m")
        index = getattr(mdata, f"{attr}_names")
        return np.concatenate(
            [np.atleast_1d(attrm[key][np.nonzero(index == name)[0]]) for name in names]
        )

    def test_update_simple(self, mdata, axis):
        """
        Update should work when
        - obs_names are the same across modalities,
        - var_names are unique to each modality
        """
        attr = "obs" if axis == 0 else "var"
        oattr = "var" if axis == 0 else "obs"

        for mod in mdata.mod.keys():
            assert mdata.obsmap[mod].dtype.kind == "u"
            assert mdata.varmap[mod].dtype.kind == "u"

        # names along non-axis are concatenated
        assert mdata.shape[1 - axis] == sum(mod.shape[1 - axis] for mod in mdata.mod.values())
        assert (
            getattr(mdata, f"{oattr}_names")
            == reduce(
                lambda x, y: x.append(y),
                (getattr(mod, f"{oattr}_names") for mod in mdata.mod.values()),
            )
        ).all()

        # names along axis are unioned
        axisnames = reduce(
            lambda x, y: x.union(y, sort=False),
            (getattr(mod, f"{attr}_names") for mod in mdata.mod.values()),
        )
        assert mdata.shape[axis] == axisnames.shape[0]
        assert (getattr(mdata, f"{attr}_names").sort_values() == axisnames.sort_values()).all()

        # guards against Pandas scrambling the order. This was the case for pandas < 1.4.0 when using pd.concat with an outer join on a MultiIndex.
        # reprex:
        #
        # import numpy as np
        # import pandas as pd
        # df1 = pd.DataFrame({"a": np.repeat(np.arange(5), 2), "b": np.tile(np.asarray([0, 1]), 5), "c": np.arange(10)}).set_index("a").set_index("b", append=True)
        # df2 = pd.DataFrame({"a": np.repeat(np.arange(10), 2), "b": np.tile(np.asarray([0, 1]), 10), "d": np.arange(20)}).set_index("a").set_index("b", append=True)
        # df1 = df1.iloc[::-1, :]
        # df = pd.concat((kdf1, df2), axis=1, join="outer", sort=False)
        assert (
            getattr(mdata, f"{attr}_names")[: mdata["mod1"].shape[axis]]
            == getattr(mdata["mod1"], f"{attr}_names")
        ).all()

    def test_update_add_modality(self, modalities, axis):
        modnames = list(modalities.keys())
        mdata = MuData({modname: modalities[modname] for modname in modnames[:-2]}, axis=axis)

        attr = "obs" if axis == 0 else "var"
        oattr = "var" if axis == 0 else "obs"

        for i in (-2, -1):
            old_attrnames = getattr(mdata, f"{attr}_names")
            old_oattrnames = getattr(mdata, f"{oattr}_names")

            some_obs_names = mdata.obs_names[:2]
            mdata.obsm["test"] = np.random.normal(size=(mdata.n_obs, 1))
            true_obsm_values = self.get_attrm_values(mdata, "obs", "test", some_obs_names)

            mdata.mod[modnames[i]] = modalities[modnames[i]]
            mdata.update()

            for mod in mdata.mod.keys():
                assert mdata.obsmap[mod].dtype.kind == "u"
                assert mdata.varmap[mod].dtype.kind == "u"

            test_obsm_values = self.get_attrm_values(mdata, "obs", "test", some_obs_names)
            if axis == 1:
                assert np.isnan(mdata.obsm["test"]).sum() == modalities[modnames[i]].n_obs
                assert np.all(np.isnan(mdata.obsm["test"][-modalities[modnames[i]].n_obs :]))
                assert np.all(~np.isnan(mdata.obsm["test"][: -modalities[modnames[i]].n_obs]))
                assert (
                    test_obsm_values[~np.isnan(test_obsm_values)].reshape(-1)
                    == true_obsm_values.reshape(-1)
                ).all()
            else:
                assert (test_obsm_values == true_obsm_values).all()

            attrnames = getattr(mdata, f"{attr}_names")
            oattrnames = getattr(mdata, f"{oattr}_names")
            assert (attrnames[: old_attrnames.size] == old_attrnames).all()
            assert (oattrnames[: old_oattrnames.size] == old_oattrnames).all()

            assert (
                attrnames
                == old_attrnames.union(
                    getattr(modalities[modnames[i]], f"{attr}_names"), sort=False
                )
            ).all()
            assert (
                oattrnames
                == old_oattrnames.append(getattr(modalities[modnames[i]], f"{oattr}_names"))
            ).all()

    def test_update_delete_modality(self, mdata, axis):
        modnames = list(mdata.mod.keys())
        attr = "obs" if axis == 0 else "var"
        oattr = "var" if axis == 0 else "obs"
        attrm = f"{attr}m"
        oattrm = f"{oattr}m"

        fullbatch = getattr(mdata, attr)["batch"]
        fullobatch = getattr(mdata, oattr)["batch"]
        fulltestm = getattr(mdata, attrm)["test"]
        fullotestm = getattr(mdata, oattrm)["test"]
        keptmask = (getattr(mdata, f"{attr}map")[modnames[1]].reshape(-1) > 0) | (
            getattr(mdata, f"{attr}map")[modnames[2]].reshape(-1) > 0
        )
        keptomask = (getattr(mdata, f"{oattr}map")[modnames[1]].reshape(-1) > 0) | (
            getattr(mdata, f"{oattr}map")[modnames[2]].reshape(-1) > 0
        )

        del mdata.mod[modnames[0]]
        mdata.update()

        for mod in mdata.mod.keys():
            assert mdata.obsmap[mod].dtype.kind == "u"
            assert mdata.varmap[mod].dtype.kind == "u"

        assert mdata.shape[1 - axis] == sum(mod.shape[1 - axis] for mod in mdata.mod.values())
        assert (getattr(mdata, attr)["batch"] == fullbatch[keptmask]).all()
        assert (getattr(mdata, oattr)["batch"] == fullobatch[keptomask]).all()
        assert (getattr(mdata, attrm)["test"] == fulltestm[keptmask, :]).all()
        assert (getattr(mdata, oattrm)["test"] == fullotestm[keptomask, :]).all()

        fullbatch = getattr(mdata, attr)["batch"]
        fullobatch = getattr(mdata, oattr)["batch"]
        fulltestm = getattr(mdata, attrm)["test"]
        fullotestm = getattr(mdata, oattrm)["test"]
        keptmask = getattr(mdata, f"{attr}map")[modnames[1]].reshape(-1) > 0
        keptomask = getattr(mdata, f"{oattr}map")[modnames[1]].reshape(-1) > 0

        del mdata.mod[modnames[2]]
        mdata.update()

        assert mdata.shape[1 - axis] == sum(mod.shape[1 - axis] for mod in mdata.mod.values())
        assert (getattr(mdata, oattr)["batch"] == fullobatch[keptomask]).all()
        assert (getattr(mdata, attr)["batch"] == fullbatch[keptmask]).all()
        assert (getattr(mdata, attrm)["test"] == fulltestm[keptmask, :]).all()
        assert (getattr(mdata, oattrm)["test"] == fullotestm[keptomask, :]).all()

    def test_update_intersecting(self, modalities, axis):
        """
        Update should work when
        - obs_names are the same across modalities,
        - there are intersecting var_names,
          which are unique in each modality
        """
        attr = "obs" if axis == 0 else "var"
        oattr = "var" if axis == 0 else "obs"
        for m, mod in modalities.items():
            setattr(
                mod,
                f"{oattr}_names",
                [
                    f"{m}_{oattr}{j}" if j != 0 else f"{oattr}_{j}"
                    for j in range(mod.shape[1 - axis])
                ],
            )

        mdata = MuData(modalities, axis=axis)

        for mod in mdata.mod.keys():
            assert mdata.obsmap[mod].dtype.kind == "u"
            assert mdata.varmap[mod].dtype.kind == "u"

        # names along non-axis are concatenated
        assert mdata.shape[1 - axis] == sum(mod.shape[1 - axis] for mod in modalities.values())
        assert (
            getattr(mdata, f"{oattr}_names")
            == reduce(
                lambda x, y: x.append(y),
                (getattr(mod, f"{oattr}_names") for mod in modalities.values()),
            )
        ).all()

        # names along axis are unioned
        axisnames = reduce(
            lambda x, y: x.union(y, sort=False),
            (getattr(mod, f"{attr}_names") for mod in modalities.values()),
        )
        assert mdata.shape[axis] == axisnames.shape[0]
        assert (getattr(mdata, f"{attr}_names") == axisnames).all()

    def test_update_after_filter_obs_adata(self, mdata, axis):
        """
        Check for muon issue #44.
        """
        # Replicate in-place filtering in muon:
        # mu.pp.filter_obs(mdata['mod1'], 'min_count', lambda x: (x < -2))

        old_obsnames = mdata.obs_names
        old_varnames = mdata.var_names

        filtermask = mdata["mod3"].obs["min_count"] < -2
        fullfiltermask = mdata.obsmap["mod3"].copy() > 0
        fullfiltermask[fullfiltermask] = filtermask
        keptmask = (mdata.obsmap["mod1"] > 0) | (mdata.obsmap["mod2"] > 0) | fullfiltermask

        some_obs_names = mdata[keptmask, :].obs_names.values[:2]
        true_obsm_values = self.get_attrm_values(mdata[keptmask], "obs", "test", some_obs_names)

        mdata.mod["mod3"] = mdata["mod3"][mdata["mod3"].obs["min_count"] < -2].copy()
        mdata.update()

        for mod in mdata.mod.keys():
            assert mdata.obsmap[mod].dtype.kind == "u"
            assert mdata.varmap[mod].dtype.kind == "u"

        assert mdata.obs["batch"].isna().sum() == 0

        assert (mdata.var_names == old_varnames).all()
        if axis == 0:
            # check if the order is preserved
            assert (mdata.obs_names == old_obsnames[old_obsnames.isin(mdata.obs_names)]).all()

        test_obsm_values = self.get_attrm_values(mdata, "obs", "test", some_obs_names)
        assert (true_obsm_values == test_obsm_values).all()

    def test_update_after_obs_reordered(self, mdata):
        """
        Update should work if obs are reordered.
        """
        some_obs_names = mdata.obs_names.values[:2]

        true_obsm_values = self.get_attrm_values(mdata, "obs", "test", some_obs_names)

        mdata.mod["mod1"] = mdata["mod1"][::-1].copy()
        mdata.update()

        for mod in mdata.mod.keys():
            assert mdata.obsmap[mod].dtype.kind == "u"
            assert mdata.varmap[mod].dtype.kind == "u"

        test_obsm_values = self.get_attrm_values(mdata, "obs", "test", some_obs_names)

        assert (true_obsm_values == test_obsm_values).all()


@pytest.mark.usefixtures("filepath_h5mu")
class TestMuDataLegacy:
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

    @pytest.mark.parametrize("obs_mod", ["unique", "extreme_duplicated"])
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

    @pytest.mark.parametrize("obs_mod", ["unique", "extreme_duplicated"])
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
    def test_update_after_filter_obs_adata(self, mdata_legacy):
        """
        Check for muon issue #44.
        """
        # Replicate in-place filtering in muon:
        # mu.pp.filter_obs(mdata['mod1'], 'min_count', lambda x: (x < -2))
        mdata_legacy.mod["mod1"] = mdata_legacy["mod1"][
            mdata_legacy["mod1"].obs["min_count"] < -2
        ].copy()
        old_obsnames = mdata_legacy.obs_names
        mdata_legacy.update()
        assert mdata_legacy.obs["batch"].isna().sum() == 0

    @pytest.mark.parametrize("obs_mod", ["unique", "extreme_duplicated"])
    @pytest.mark.parametrize("obs_across", ["intersecting"])
    @pytest.mark.parametrize("obs_n", ["joint", "disjoint"])
    def test_update_after_obs_reordered(self, mdata_legacy):
        """
        Update should work if obs are reordered.
        """
        mdata_legacy.obsm["test_obsm"] = np.random.normal(size=(mdata_legacy.n_obs, 2))

        some_obs_names = mdata_legacy.obs_names.values[:2]

        true_obsm_values = [
            mdata_legacy.obsm["test_obsm"][np.where(mdata_legacy.obs_names.values == name)[0][0]]
            for name in some_obs_names
        ]

        mdata_legacy.mod["mod1"] = mdata_legacy["mod1"][::-1].copy()
        mdata_legacy.update()

        test_obsm_values = [
            mdata_legacy.obsm["test_obsm"][np.where(mdata_legacy.obs_names == name)[0][0]]
            for name in some_obs_names
        ]

        assert all(
            [all(true_obsm_values[i] == test_obsm_values[i]) for i in range(len(true_obsm_values))]
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
