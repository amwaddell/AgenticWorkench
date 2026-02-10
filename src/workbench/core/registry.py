"""
Component registry for building components from configuration.

This implements a factory pattern that maps config to component instances.
"""

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class ComponentRegistry:
    """
    Registry for component constructors.

    Maps component names to their constructor functions.
    Allows building components dynamically from configuration.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._constructors: dict[str, Callable[..., Any]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        constructor: Callable[..., T],
        component_type: str | None = None,
        description: str | None = None,
    ) -> None:
        """
        Register a component constructor.

        Args:
            name: Unique name for this component (e.g., "llamacpp_server")
            constructor: Function that builds the component
            component_type: Type of component (e.g., "language_model", "retriever")
            description: Human-readable description

        Raises:
            ValueError: If name already registered
        """
        if name in self._constructors:
            raise ValueError(f"Component already registered: {name}")

        self._constructors[name] = constructor
        self._metadata[name] = {
            "component_type": component_type,
            "description": description,
        }

    def unregister(self, name: str) -> None:
        """
        Unregister a component.

        Args:
            name: Component name to unregister
        """
        self._constructors.pop(name, None)
        self._metadata.pop(name, None)

    def build(self, name: str, **kwargs: Any) -> Any:
        """
        Build a component instance.

        Args:
            name: Name of registered component
            **kwargs: Arguments to pass to constructor

        Returns:
            Component instance

        Raises:
            KeyError: If component not registered
        """
        if name not in self._constructors:
            raise KeyError(
                f"Component not registered: {name}. "
                f"Available: {list(self._constructors.keys())}"
            )

        constructor = self._constructors[name]
        return constructor(**kwargs)

    def build_from_config(self, config: dict[str, Any]) -> Any:
        """
        Build a component from a config dictionary.

        Expected config format:
        {
            "provider": "component_name",
            "param1": "value1",
            ...
        }

        Args:
            config: Configuration dictionary with "provider" key

        Returns:
            Component instance

        Raises:
            ValueError: If config missing "provider" key
            KeyError: If provider not registered
        """
        if "provider" not in config:
            raise ValueError("Config must include 'provider' key")

        provider = config["provider"]
        # Extract all config except "provider" to pass as kwargs
        kwargs = {k: v for k, v in config.items() if k != "provider"}

        return self.build(provider, **kwargs)

    def list_components(
        self, component_type: str | None = None
    ) -> list[dict[str, Any]]:
        """
        List registered components.

        Args:
            component_type: Optional filter by component type

        Returns:
            List of component info dictionaries
        """
        components = []
        for name, metadata in self._metadata.items():
            if (
                component_type is None
                or metadata.get("component_type") == component_type
            ):
                components.append(
                    {
                        "name": name,
                        **metadata,
                    }
                )
        return components

    def is_registered(self, name: str) -> bool:
        """
        Check if a component is registered.

        Args:
            name: Component name

        Returns:
            True if registered, False otherwise
        """
        return name in self._constructors

    def clear(self) -> None:
        """Clear all registered components."""
        self._constructors.clear()
        self._metadata.clear()


# Global registry instance
# Components will register themselves on import
_global_registry = ComponentRegistry()


def get_registry() -> ComponentRegistry:
    """
    Get the global component registry.

    Returns:
        Global ComponentRegistry instance
    """
    return _global_registry


def register_component(
    name: str,
    component_type: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to register a component constructor.

    Usage:
        @register_component("my_component", component_type="retriever")
        def build_my_component(**kwargs):
            return MyComponent(**kwargs)

    Args:
        name: Component name
        component_type: Type of component
        description: Description

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        _global_registry.register(
            name=name,
            constructor=func,
            component_type=component_type,
            description=description,
        )
        return func

    return decorator
