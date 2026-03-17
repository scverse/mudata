import numpy as np

import mudata

N1 = 15


def test_merge(mdata_nouniqueobs):
    mdata1, mdata2 = mdata_nouniqueobs[:N1, :].copy(), mdata_nouniqueobs[N1:, :].copy()
    mdata_ = mudata.concat([mdata1, mdata2])
    assert list(mdata_.mod.keys()) == ["mod1", "mod2"]
    for m in mdata_.mod_names:
        assert mdata_.mod[m].shape == mdata_nouniqueobs.mod[m].shape
    assert np.array_equal(mdata_.mod["mod1"].X, mdata_nouniqueobs.mod["mod1"].X)
    assert np.array_equal(mdata_.mod["mod2"].X, mdata_nouniqueobs.mod["mod2"].X)


def test_merge_and_write(mdata_nouniqueobs, filepath_h5mu):
    mdata1, mdata2 = mdata_nouniqueobs[:N1, :].copy(), mdata_nouniqueobs[N1:, :].copy()
    mdata_merged = mudata.concat([mdata1, mdata2])
    mdata_merged.write_h5mu(filepath_h5mu)
    mdata_ = mudata.read_h5mu(filepath_h5mu)
    assert list(mdata_.mod.keys()) == ["mod1", "mod2"]
    for m in mdata_.mod_names:
        assert mdata_.mod[m].shape == mdata_nouniqueobs.mod[m].shape
    assert np.array_equal(mdata_.mod["mod1"].X, mdata_nouniqueobs.mod["mod1"].X)
    assert np.array_equal(mdata_.mod["mod2"].X, mdata_nouniqueobs.mod["mod2"].X)
