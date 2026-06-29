import json
from dataclasses import fields
from importlib import metadata, resources
from urllib.request import urlopen

import anndata as ad
import jsonschema
import numpy as np
import pandas as pd
import pytest
import referencing
from packaging.version import Version

import mudata as md

if Version(metadata.version("anndata")) < Version("0.13dev0"):
    pytest.skip("anndata version too old, no accessor support", allow_module_level=True)

from mudata.acc import A


@pytest.fixture
def mdata_augmented(mdata: md.MuData, rng: np.random.Generator):
    mdata["mod1"].layers["counts"] = rng.poisson(1, size=mdata["mod1"].shape)
    mdata["mod2"].varm["test"] = rng.normal(size=(mdata["mod2"].n_vars, 3))
    mdata["mod2"].obsp["test"] = rng.normal(size=(mdata["mod2"].n_obs, mdata["mod2"].n_obs))

    return mdata


@pytest.fixture(scope="session")
def mudata_json_schema():
    schema = json.loads(resources.files(md).joinpath("acc/acc-schema-v1.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    return schema


@pytest.fixture(scope="session")
def anndata_schema_registry():
    anndata_schema_uri = "https://anndata.scverse.org/en/latest/acc-schema-v1.json"
    with urlopen(anndata_schema_uri) as response:
        anndata_schema = json.load(response)
    schema = referencing.Resource.from_contents(anndata_schema)
    return referencing.Registry().with_resource(anndata_schema_uri, schema)


def test_anndata_accessors(mdata: md.MuData):
    assert ad.acc.A.obs["arange"] in mdata
    assert (mdata[ad.acc.A.obs["arange"]] == mdata.obs["arange"]).all()
    with pytest.raises(KeyError, match="test"):
        mdata[ad.acc.A.var["test"]]
    with pytest.raises(KeyError, match="there is one in"):
        mdata[ad.acc.A.var["assert-bool"]]


PATHS = [
    (A.mod["mod1"], lambda md: md.mod["mod1"]),
    (A["mod1"], lambda md: md["mod1"]),
    (A.mod["mod1"].var, lambda md: md.mod["mod1"].var),
    (A.mod["mod1"].var["assert-bool"], lambda md: md.mod["mod1"].var["assert-bool"]),
    (A.mod["mod1"].X, lambda md: md.mod["mod1"].X),
    (A.mod["mod1"].X["obs_2", :], lambda md: md.mod["mod1"]["obs_2", :].X.squeeze()),
    (A.mod["mod1"].X[:, "mod1_var_1"], lambda md: md.mod["mod1"][:, "mod1_var_1"].X.squeeze()),
    (A["mod1"].layers, lambda md: md["mod1"].layers),
    (A["mod1"].layers["counts"], lambda md: md["mod1"].layers["counts"]),
    (A["mod1"].layers["counts"]["obs_2", :], lambda md: md["mod1"]["obs_2", :].layers["counts"].squeeze()),
    (A["mod2"].varm, lambda md: md["mod2"].varm),
    (A["mod2"].varm["test"], lambda md: md["mod2"].varm["test"]),
    (A["mod2"].obsp, lambda md: md["mod2"].obsp),
    (A["mod2"].obsp["test"], lambda md: md["mod2"].obsp["test"]),
    (A["mod2"].obsp["test"][:, "obs_3"], lambda md: md["mod2"].obsp["test"][:, md["mod2"].obs_names.get_loc("obs_3")]),
    (A.obsmap, lambda md: md.obsmap),
    (A.varmap, lambda md: md.varmap),
    (A.obsmap["mod1"], lambda md: md.obsmap["mod1"]),
    (A.varmap["mod2"], lambda md: md.varmap["mod2"]),
]


@pytest.mark.parametrize("acc", [path[0] for path in PATHS])
def test_in(mdata_augmented: md.MuData, acc):
    assert acc in mdata_augmented


@pytest.mark.parametrize(
    "acc",
    [
        A.mod["mod3"],
        A["mod3"],
        A.mod["mod3"].var,
        A.mod["mod1"].var["does_not_exist"],
        A.mod["mod3"].X,
        A.mod["mod3"].X["obs_2", :],
        A.mod["mod3"].X[:, "mod1_var_1"],
        A["mod2"].layers,
        A["mod1"].layers["does_not_exist"],
        A["mod1"].layers["does_not_exist"]["obs_2", :],
        A["mod1"].obsp,
        A["mod2"].obsp["does_not_exist"],
        A["mod2"].obsp["does_not_exist"][:, "obs_3"],
        A.obsmap["mod3"],
        A.varmap["mod3"],
    ],
)
def test_not_in(mdata: md.MuData, acc):
    assert acc not in mdata


@pytest.mark.parametrize("acc_expected", [path for path in PATHS if isinstance(path[0], ad.acc.AdRef)])
def test_get(mdata_augmented: md.MuData, acc_expected):
    acc, expected = acc_expected

    val = mdata_augmented[acc]
    expected = expected(mdata_augmented)
    if isinstance(expected, pd.DataFrame | pd.Series | np.ndarray):
        assert np.all(val == expected)
    else:
        assert val == expected


def test_no_data():
    with pytest.raises(AttributeError):
        A.X  # noqa: B018
    with pytest.raises(AttributeError):
        A.layers  # noqa: B018

    for field in fields(A):
        assert field.name not in ("X", "layers")


def test_resolve():
    assert A.resolve("mod.rna.X[:, ACT1]") == A.mod["rna"].X[:, "ACT1"]
    assert A.resolve("obsmap.rna") == A.obsmap["rna"]

    with pytest.raises(ValueError, match="Unknown accessor"):
        A.resolve("rna.X[:, :]")

    with pytest.raises(ValueError, match="empty modality"):
        A.resolve("mod..X[:, :]")

    with pytest.raises(ValueError, match="period-separated"):
        A.resolve("abcd")


@pytest.mark.parametrize("acc", [path[0] for path in PATHS if isinstance(path[0], ad.acc.AdRef)])
def test_to_from_json(mudata_json_schema, anndata_schema_registry, acc):
    serialized = A.to_json(acc)
    jsonschema.validate(serialized, mudata_json_schema, registry=anndata_schema_registry)
    if isinstance(acc.acc, md.acc.ModMapAcc):
        assert serialized[0] == f"{acc.acc.dim}map"
        assert serialized[1] == acc.idx
    else:
        assert serialized[0] == "mod"
        assert serialized[1] == acc.acc.mod
        assert serialized[2] == ad.acc.A.to_json(acc)

    assert A.from_json(serialized) == acc


@pytest.mark.parametrize(
    "acc",
    [path[0] for path in PATHS if isinstance(path[0], ad.acc.AdRef) and not isinstance(path[0].acc, md.acc.ModMapAcc)],
)
def test_to_from_json_mod(mudata_json_schema, anndata_schema_registry, acc):
    modA = A.mod["foobar"]

    serialized = modA.to_json(acc)
    jsonschema.validate(serialized, mudata_json_schema, registry=anndata_schema_registry)
    assert serialized[0] == "mod"
    assert serialized[1] == "foobar"
    assert serialized[2] == ad.acc.A.to_json(acc)

    assert A.from_json(serialized).acc.mod == "foobar"
