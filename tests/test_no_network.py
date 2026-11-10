"""The library promises to send nothing anywhere.

Sockets are disabled for the whole test run (see ``addopts`` in pyproject.toml), so any
code path that tries to reach the network fails its test. These tests check that the
guard is really active and that importing the package does not trip it.
"""

import importlib
import socket
import sys
import warnings

import pytest
from pytest_socket import SocketBlockedError


def test_network_guard_is_active() -> None:
    with warnings.catch_warnings():
        # The guard warns and then raises. Only the refusal matters here.
        warnings.simplefilter("ignore")
        with pytest.raises(SocketBlockedError):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_import_opens_no_connection() -> None:
    for name in [m for m in sys.modules if m == "obscuradp" or m.startswith("obscuradp.")]:
        del sys.modules[name]
    importlib.import_module("obscuradp")  # would raise SocketBlockedError on any connection
