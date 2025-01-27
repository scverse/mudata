# Archives with legacy test data

These archives should contain folders containing `.h5mu` test files
creasted with older `mudata` versions for regression testing.

## Format

A folder with the format `v<version>` indicating the `mudata` version used
to create the file.

Containing:
- `create_testfiles.py`: a script creating the testfiles. Should contain a
    uv header specifying the critical dependencies. Can be run with `uv run create_testfile.py`
    to create the testfile(s)
- `mudata.h5mu`: the created testfile. Needs to contain at minimum the data expected by
    `test_io_backwards_compat.py`. 

## Automated tests

The test `test_io_backwards_compat.py` will check each `mudata.h5mu` file in this folder if it can
be loaded correctly by the current `mudata` version.