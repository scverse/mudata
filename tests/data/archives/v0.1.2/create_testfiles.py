# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mudata==0.1.2",
#     "numpy==1.*", # required as numpy 2 raises errors
# ]
# [tool.uv]
# exclude-newer = "2024-12-01T00:00:00Z"
# ///
import os

import numpy as np
from anndata import AnnData

from mudata import MuData


def create_mudata() -> MuData:
    """Create a mudata testfile

    Returns:
        MuData: MuData object with two AnnData objects
    """
    return MuData(
        {
            "mod1": AnnData(np.arange(0, 100, 0.1).reshape(-1, 10)),
            "mod2": AnnData(np.arange(101, 2101, 1).reshape(-1, 20)),
        }
    )


def write_h5mu_testfile(mdata: MuData, filepath_h5mu: str | os.PathLike) -> None:
    """Write a MuData object to a file

    Args:
        mdata (MuData): MuData object to write
        filepath_h5mu (str | os.PathLike): Path to write the MuData object

    Returns:
        None
    """
    mdata.write(filepath_h5mu)


if __name__ == "__main__":
    mdata = create_mudata()
    write_h5mu_testfile(mdata, "mudata.h5mu")
