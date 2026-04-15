import importlib

import pytest

import opensportslib


def test_lazy_attributes_are_exposed():
    expected_modules = {
        "model": "opensportslib.apis",
        "metrics": "opensportslib.metrics",
        "datasets": "opensportslib.datasets",
        "core": "opensportslib.core",
    }
    for attr, expected in expected_modules.items():
        module = getattr(opensportslib, attr)
        assert module.__name__ == expected
        assert importlib.import_module(module.__name__) is module


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        getattr(opensportslib, "unknown_api")
