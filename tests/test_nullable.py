import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from mudata import MuData

# Dimensions
N = 100
D1, D2 = 10, 20
D = D1 + D2


@pytest.fixture()
def mdata():
    mod1 = AnnData(np.arange(0, 100, 0.1).reshape(-1, D1))
    mod1.obs_names = [f"obs{i}" for i in range(mod1.n_obs)]
    mod1.var_names = [f"var{i}" for i in range(D1)]

    mod2 = AnnData(np.arange(3101, 5101, 1).reshape(-1, D2))
    mod2.obs_names = mod1.obs_names.copy()
    mod2.var_names = [f"var{i}" for i in range(D1, D)]

    # bool + bool -> bool
    mod1.var["assert-bool"] = True
    mod2.var["assert-bool"] = False

    # bool + NA -> boolean
    mod1.var["assert-boolean-1"] = True
    mod2.var["assert-boolean-2"] = False

    mdata = MuData({"mod1": mod1, "mod2": mod2})
    return mdata


class TestMuData:
    def test_mdata_bool_boolean(self, mdata):
        assert mdata.var["assert-bool"].dtype == bool
        assert isinstance(mdata.var["mod1:assert-boolean-1"].dtype, pd.BooleanDtype)
        assert isinstance(mdata.var["mod2:assert-boolean-2"].dtype, pd.BooleanDtype)
