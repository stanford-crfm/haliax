from typing import TypeVar

import jax
import pytest


T = TypeVar("T")


def skip_if_not_enough_devices(count: int):
    return pytest.mark.skipif(len(jax.devices()) < count, reason=f"Not enough devices ({len(jax.devices())})")


def has_torch():
    try:
        import torch  # noqa F401

        return True
    except ImportError:
        return False


def skip_if_no_torch(f):
    return pytest.mark.skipif(not has_torch(), reason="torch not installed")(f)
