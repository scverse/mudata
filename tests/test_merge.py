import numpy as np
import pytest

import mudata as md


@pytest.mark.parametrize("roundtrip", (False, True))
def test_merge(mdata_nouniqueobs, filepath_h5mu, roundtrip):
    mdata1, mdata2 = mdata_nouniqueobs[:15, :].copy(), mdata_nouniqueobs[15:, :].copy()
    mdata_ = md.concat([mdata1, mdata2])
    if roundtrip:
        mdata_.write_h5mu(filepath_h5mu)
        mdata_ = md.read_h5mu(filepath_h5mu)
    assert list(mdata_.mod.keys()) == ["mod1", "mod2"]
    for m in mdata_.mod_names:
        assert mdata_.mod[m].shape == mdata_nouniqueobs.mod[m].shape
    assert np.array_equal(mdata_.mod["mod1"].X, mdata_nouniqueobs.mod["mod1"].X)
    assert np.array_equal(mdata_.mod["mod2"].X, mdata_nouniqueobs.mod["mod2"].X)
