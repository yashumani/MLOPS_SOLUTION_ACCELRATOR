"""Test seed_control utility."""

import os
import random

import numpy as np

from utils.seed_control import set_global_seed, resolve_seed


def test_set_global_seed_returns_seed():
    assert set_global_seed(123) == 123


def test_set_global_seed_makes_random_deterministic():
    set_global_seed(7)
    a = [random.random() for _ in range(5)]
    set_global_seed(7)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_global_seed_makes_numpy_deterministic():
    set_global_seed(11)
    a = np.random.rand(5).tolist()
    set_global_seed(11)
    b = np.random.rand(5).tolist()
    assert a == b


def test_set_global_seed_sets_python_hash_seed():
    set_global_seed(99)
    assert os.environ["PYTHONHASHSEED"] == "99"


def test_resolve_seed_from_cfg():
    assert resolve_seed({"random_seed": 7}) == 7


def test_resolve_seed_default_when_missing():
    assert resolve_seed({}) == 42
    assert resolve_seed(None) == 42


def test_resolve_seed_handles_invalid_value():
    assert resolve_seed({"random_seed": "notanint"}) == 42
