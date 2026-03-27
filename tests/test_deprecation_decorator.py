import pytest

from mudata._core.utils import deprecated


@pytest.fixture(params=[None, "Test message."])
def msg(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(
    params=[
        None,
        "Test function",
        """Test function

    This is a test.

    Parameters
    ----------
    foo
        bar
    bar
        baz
""",
    ]
)
def docstring(request):
    return request.param


@pytest.fixture
def deprecated_func(msg, docstring):
    def func(foo, bar):
        return 42

    func.__doc__ = docstring
    return deprecated(version="foo", msg=msg)(func)


def test_deprecation_decorator(deprecated_func, docstring, msg):
    with pytest.warns(FutureWarning, match="deprecated"):
        assert deprecated_func(1, 2) == 42

    lines = deprecated_func.__doc__.expandtabs().splitlines()
    if docstring is None:
        assert lines[0].startswith(".. version-deprecated::")
    else:
        lines_orig = docstring.expandtabs().splitlines()
        assert lines[0] == lines_orig[0]
        assert len(lines[1].strip()) == 0
        if len(lines_orig) == 1:
            assert lines[2].startswith(".. version-deprecated")
            if msg is not None:
                assert lines[3] == f"   {msg}"
        else:
            assert lines[2].startswith("    .. version-deprecated")
            if msg is not None:
                assert lines[3] == f"       {msg}"
