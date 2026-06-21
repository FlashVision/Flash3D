"""Flash3D Registry – Centralized component registration for models, renderers, datasets, and tasks."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type


class Registry:
    """A universal registry that maps string names to classes or factory functions.

    Supports decorator-based and explicit registration.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def registered_names(self) -> list[str]:
        return list(self._registry.keys())

    def register(self, name: Optional[str] = None) -> Callable:
        """Decorator to register a class or function.

        Usage:
            @MODELS.register("gaussian_splatting")
            class GaussianSplatting: ...
        """

        def decorator(cls_or_fn: Any) -> Any:
            key = name if name is not None else cls_or_fn.__name__
            if key in self._registry:
                raise KeyError(f"'{key}' is already registered in {self._name}")
            self._registry[key] = cls_or_fn
            return cls_or_fn

        return decorator

    def register_module(self, name: str, module: Any) -> None:
        """Explicitly register a module by name."""
        if name in self._registry:
            raise KeyError(f"'{name}' is already registered in {self._name}")
        self._registry[name] = module

    def get(self, name: str) -> Any:
        """Retrieve a registered component by name."""
        if name not in self._registry:
            raise KeyError(
                f"'{name}' not found in {self._name}. "
                f"Available: {self.registered_names}"
            )
        return self._registry[name]

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Build (instantiate) a registered component."""
        cls_or_fn = self.get(name)
        return cls_or_fn(*args, **kwargs)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={self.registered_names})"


MODELS = Registry("models")
RENDERERS = Registry("renderers")
DATASETS = Registry("datasets")
TASKS = Registry("tasks")
