import numpy as np
from anndata import AnnData

from mudata import MuData, to_mudata


def test_to_anndata_simple(mdata):
    adata = mdata.to_anndata()
    assert isinstance(adata, AnnData)
    assert adata.shape == mdata.shape
    assert "arange" in adata.obs.columns
    assert np.array_equal(adata.obs["arange"], np.arange(mdata.n_obs))


def test_to_mudata_simple(mdata):
    adata = mdata.to_anndata()
    mdata_from_adata = to_mudata(adata, axis=0, by="feature_type")
    assert isinstance(mdata_from_adata, MuData)
    assert mdata_from_adata.shape == adata.shape
    assert mdata_from_adata.shape == mdata.shape


def test_from_anndata(rng):
    adata = AnnData(np.arange(0, 300, 0.1).reshape(-1, 30))
    adata.var["feature_types"] = rng.choice(["mod1", "mod2", "mod3"], size=adata.n_vars)
    mdata = MuData(adata)
    assert mdata.n_mod == 3
    assert sorted(mdata.mod_names) == ["mod1", "mod2", "mod3"]
    for m, mod in mdata.mod.items():
        assert (mod.var_names == adata.var_names[adata.var.feature_types == m]).all()
