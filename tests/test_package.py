import re


def test_import_package_and_version():
    import era_v4

    assert hasattr(era_v4, "__version__")
    assert re.match(r"^\d+\.\d+\.\d+", era_v4.__version__) is not None
