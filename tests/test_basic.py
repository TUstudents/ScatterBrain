"""Basic tests for ScatterBrain."""

from scatterbrain import __version__


def test_version():
    assert isinstance(__version__, str)
