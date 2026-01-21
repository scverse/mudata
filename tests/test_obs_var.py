import unittest

import numpy as np
import pytest
from anndata import AnnData

import mudata
from mudata import MuData


@pytest.fixture()
def mdata():
    mod1 = AnnData(np.arange(0, 100, 0.1).reshape(-1, 10))
    mod2 = AnnData(np.arange(101, 2101, 1).reshape(-1, 20))
    mods = {"mod1": mod1, "mod2": mod2}
    # Make var_names different in different modalities
    for m in ["mod1", "mod2"]:
        mods[m].var_names = [f"{m}_var{i}" for i in range(mods[m].n_vars)]
    mdata = MuData(mods)
    return mdata


@pytest.mark.usefixtures("filepath_h5mu")
class TestMuData:
    def test_obs_global_columns(self, mdata, filepath_h5mu):
        for m, mod in mdata.mod.items():
            mod.obs["demo"] = m
        mdata.obs["demo"] = "global"
        mdata.update()
        assert list(mdata.obs.columns.values) == [
            f"{m}:demo" for m in mdata.mod.keys()
        ] + ["demo"]
        mdata.write(filepath_h5mu)
        mdata_ = mudata.read(filepath_h5mu)
        assert list(mdata_.obs.columns.values) == [
            f"{m}:demo" for m in mdata_.mod.keys()
        ] + ["demo"]

    def test_var_global_columns(self, mdata, filepath_h5mu):
        for m, mod in mdata.mod.items():
            mod.var["demo"] = m
        mdata.update()
        mdata.var["global"] = "global_var"
        mdata.update()
        assert list(mdata.var.columns.values) == ["demo", "global"]
        del mdata.var["global"]
        mdata.update()
        assert list(mdata.var.columns.values) == ["demo"]
        mdata.write(filepath_h5mu)
        mdata_ = mudata.read(filepath_h5mu)
        assert list(mdata_.var.columns.values) == ["demo"]


class TestObsVarNamesRename:
    """Test obs_names and var_names renaming with subset modalities (issue #112)."""

    def test_obs_names_rename_with_subset_modality(self):
        """Test that renaming global obs_names correctly updates subset modality obs_names."""
        import pandas as pd

        # Full modality with 4 cells
        gex = AnnData(
            X=np.random.rand(4, 3),
            obs=pd.DataFrame(index=["cellA-1", "cellB-1", "cellC-1", "cellD-1"]),
            var=pd.DataFrame(index=["G1", "G2", "G3"]),
        )
        # Subset modality with only 2 cells (cellA-1 and cellC-1)
        airr = AnnData(
            X=np.empty((2, 0)),
            obs=pd.DataFrame(index=["cellA-1", "cellC-1"]),
        )
        mdata = MuData({"gex": gex, "airr": airr})

        # Rename global obs_names by removing suffix
        mdata.obs_names = [s.split("-", 1)[0] for s in mdata.obs_names]

        # Verify global and full modality
        assert list(mdata.obs_names) == ["cellA", "cellB", "cellC", "cellD"]
        assert list(mdata["gex"].obs_names) == ["cellA", "cellB", "cellC", "cellD"]

        # Verify subset modality gets correct names (not reordered)
        assert list(mdata["airr"].obs_names) == ["cellA", "cellC"]

    def test_var_names_rename_with_subset_modality(self):
        """Test that renaming global var_names correctly updates subset modality var_names."""
        import pandas as pd

        # Full modality with 4 variables
        mod1 = AnnData(
            X=np.random.rand(2, 4),
            obs=pd.DataFrame(index=["cell1", "cell2"]),
            var=pd.DataFrame(index=["geneA-1", "geneB-1", "geneC-1", "geneD-1"]),
        )
        # Subset modality with only 2 variables (geneA-1 and geneC-1)
        mod2 = AnnData(
            X=np.random.rand(2, 2),
            obs=pd.DataFrame(index=["cell1", "cell2"]),
            var=pd.DataFrame(index=["geneA-1", "geneC-1"]),
        )
        mdata = MuData({"mod1": mod1, "mod2": mod2}, axis=1)

        # Rename global var_names by removing suffix
        mdata.var_names = [s.split("-", 1)[0] for s in mdata.var_names]

        # Verify global and full modality
        assert list(mdata.var_names) == ["geneA", "geneB", "geneC", "geneD"]
        assert list(mdata["mod1"].var_names) == ["geneA", "geneB", "geneC", "geneD"]

        # Verify subset modality gets correct names (not reordered)
        assert list(mdata["mod2"].var_names) == ["geneA", "geneC"]


if __name__ == "__main__":
    unittest.main()
