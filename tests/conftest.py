import numpy as np
import pytest


@pytest.fixture
def filepath_h5mu(tmp_path):
    return tmp_path / "testA.h5mu"


@pytest.fixture
def filepath2_h5mu(tmp_path):
    return tmp_path / "testB.h5mu"


@pytest.fixture
def filepath_zarr(tmp_path):
    return tmp_path / "testA.zarr"


@pytest.fixture
def filepath2_zarr(tmp_path):
    return tmp_path / "testB.zarr"


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)
