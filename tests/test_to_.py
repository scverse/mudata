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
