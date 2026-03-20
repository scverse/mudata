import re

modality_header_pattern = re.compile(r"^\s*(.+):\s*(\d+)\s*×\s*(\d+)\s*$")


def test_repr(mdata):
    rep = repr(mdata).splitlines()

    assert rep[0] == f"MuData object with n_obs × n_vars = {mdata.n_obs} × {mdata.n_vars}"
    assert rep[1].lstrip().startswith("obs:")
    assert rep[2].lstrip().startswith("var:")

    for col in mdata.obs.columns:
        if not any(col.startswith(f"{mod}:") for mod in mdata.mod_names):
            assert col in rep[1]
    for col in mdata.var.columns:
        if not any(col.startswith(f"{mod}:") for mod in mdata.mod_names):
            assert col in rep[2]

    assert rep[3].strip() == f"{mdata.n_mod} modalities"

    indentation = 1e6
    for line in rep[4:]:
        for i, char in enumerate(line):
            if not char.isspace():
                indentation = min(indentation, i)
    for line in rep[4:]:
        if not line[indentation].isspace():  # modality header
            match = modality_header_pattern.fullmatch(line)
            assert match is not None
            assert (cmod := match[1]) in mdata.mod
            assert int(match[2]) == mdata[cmod].n_obs
            assert int(match[3]) == mdata[cmod].n_vars
