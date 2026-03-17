import pandas as pd


def test_mdata_bool_boolean(mdata):
    assert mdata.var["assert-bool"].dtype == bool
    assert isinstance(mdata.var["mod1:assert-boolean-1"].dtype, pd.BooleanDtype)
    assert isinstance(mdata.var["mod2:assert-boolean-2"].dtype, pd.BooleanDtype)
