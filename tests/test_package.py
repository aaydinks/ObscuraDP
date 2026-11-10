"""The package installs, imports and reports a version."""

import importlib
import re
from importlib.metadata import version


def test_version_matches_metadata() -> None:
    import obscuradp

    assert obscuradp.__version__ == version("obscuradp")
    assert re.match(r"^\d+\.\d+\.\d+", obscuradp.__version__)


def test_public_names_exist() -> None:
    import obscuradp

    for name in obscuradp.__all__:
        assert hasattr(obscuradp, name)


def test_is_typed() -> None:
    from importlib.resources import files

    assert files("obscuradp").joinpath("py.typed").is_file()


def test_import_is_repeatable() -> None:
    module = importlib.import_module("obscuradp")
    assert importlib.reload(module) is module
