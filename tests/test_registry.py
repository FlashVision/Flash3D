"""Tests for Flash3D registry."""

import pytest

from flash3d.registry import Registry, MODELS, RENDERERS, DATASETS, TASKS


class TestRegistry:
    def test_register_decorator(self):
        reg = Registry("test")

        @reg.register("my_model")
        class MyModel:
            pass

        assert "my_model" in reg
        assert reg.get("my_model") is MyModel

    def test_register_module(self):
        reg = Registry("test")

        class Foo:
            pass

        reg.register_module("foo", Foo)
        assert "foo" in reg
        assert reg.get("foo") is Foo

    def test_build(self):
        reg = Registry("test")

        @reg.register("counter")
        class Counter:
            def __init__(self, start=0):
                self.value = start

        obj = reg.build("counter", start=5)
        assert obj.value == 5

    def test_duplicate_raises(self):
        reg = Registry("test")

        @reg.register("dup")
        class A:
            pass

        with pytest.raises(KeyError):
            @reg.register("dup")
            class B:
                pass

    def test_missing_raises(self):
        reg = Registry("test")
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_registered_names(self):
        reg = Registry("test")

        @reg.register("a")
        class A:
            pass

        @reg.register("b")
        class B:
            pass

        assert set(reg.registered_names) == {"a", "b"}
        assert len(reg) == 2

    def test_global_registries_exist(self):
        assert MODELS.name == "models"
        assert RENDERERS.name == "renderers"
        assert DATASETS.name == "datasets"
        assert TASKS.name == "tasks"

    def test_models_populated(self):
        # Trigger model registration by importing architectures
        from flash3d.models.architectures import gaussian_splatting, nerf, feed_forward_3dgs  # noqa: F401

        assert "gaussian_splatting" in MODELS
        assert "nerf" in MODELS
        assert "feed_forward_3dgs" in MODELS
