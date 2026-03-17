import numpy as np
import pandas as pd
import pytest

from mudata._core.utils import try_convert_dataframe_to_numpy_dtypes, try_convert_series_to_numpy_dtype


@pytest.mark.parametrize(
    ("numpy_dtype", "nullable_dtype"),
    [
        (bool, pd.BooleanDtype),
        (np.int8, pd.Int8Dtype),
        (np.uint8, pd.UInt8Dtype),
        (np.int16, pd.Int16Dtype),
        (np.uint16, pd.UInt16Dtype),
        (np.int32, pd.Int32Dtype),
        (np.uint32, pd.UInt32Dtype),
        (np.int64, pd.Int64Dtype),
        (np.uint64, pd.UInt64Dtype),
        pytest.param(np.float32, pd.Float32Dtype, marks=pytest.mark.xfail),
        pytest.param(np.float64, pd.Float64Dtype, marks=pytest.mark.xfail),
    ],
)  # pandas doesn't differentiate between NA and NaN in nullable floats, so xfail those tests.'
def test_try_convert_series_to_numpy_dtype(numpy_dtype, nullable_dtype):
    series = pd.Series(np.arange(10).astype(numpy_dtype))
    assert try_convert_series_to_numpy_dtype(series) is series

    series = series.astype(nullable_dtype())
    assert try_convert_series_to_numpy_dtype(series).dtype == numpy_dtype

    series.iloc[1] = pd.NA
    assert try_convert_series_to_numpy_dtype(series) is series


def test_try_convert_dataframe_to_numpy_dtypes():
    series = pd.Series(np.arange(10))
    df = pd.DataFrame({"numpy": series})

    series = series.astype(pd.Int64Dtype())
    df["nullable"] = series.copy()

    series.iloc[1] = pd.NA
    df["nullable_noconvert"] = series

    df = try_convert_dataframe_to_numpy_dtypes(df)
    assert df["numpy"].dtype == df["nullable"].dtype == np.int64
    assert df["nullable_noconvert"].dtype == pd.Int64Dtype()
