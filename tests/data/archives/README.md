# Archives with legacy test data

These archives should contain folders containing `.h5mu` test files
creasted with older `mudata` versions for regression testing.

## Format

A folder with the format `v<version>`.

Containing:
- `create_testfiles.py`: a script creating the testfiles. Should contain a
    uv header specifying the critical dependencies. Can be run with `uv run create_testfile.py`
    to create the testfile(s)
- `mudata.h5mu`: the created testfile