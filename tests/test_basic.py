"""Basic tests for ScatterBrain."""

import pytest
from scatterbrain import __version__

def test_version():
    assert isinstance(__version__, str)
