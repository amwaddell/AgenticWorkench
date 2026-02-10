"""
Tests for component registry.
"""

import pytest

from workbench.core.registry import ComponentRegistry, get_registry, register_component


# Fake components for testing
class FakeLanguageModel:
    """Fake language model for testing."""

    def __init__(self, model_name: str, temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, messages, **settings):
        """Fake generate method."""
        return f"Generated from {self.model_name}"


class FakeRetriever:
    """Fake retriever for testing."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def retrieve(self, query):
        """Fake retrieve method."""
        return [f"chunk_{i}" for i in range(self.top_k)]


def build_fake_model(**kwargs):
    """Constructor function for fake model."""
    return FakeLanguageModel(**kwargs)


def build_fake_retriever(**kwargs):
    """Constructor function for fake retriever."""
    return FakeRetriever(**kwargs)


def test_registry_register_and_build():
    """Test basic registration and building."""
    registry = ComponentRegistry()

    # Register a component
    registry.register("fake_model", build_fake_model, component_type="language_model")

    # Build it
    model = registry.build("fake_model", model_name="test-model", temperature=0.5)

    assert isinstance(model, FakeLanguageModel)
    assert model.model_name == "test-model"
    assert model.temperature == 0.5


def test_registry_build_unregistered_raises_error():
    """Test that building an unregistered component raises KeyError."""
    registry = ComponentRegistry()

    with pytest.raises(KeyError, match="not registered"):
        registry.build("nonexistent")


def test_registry_duplicate_registration_raises_error():
    """Test that duplicate registration raises ValueError."""
    registry = ComponentRegistry()

    registry.register("fake_model", build_fake_model)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("fake_model", build_fake_model)


def test_registry_build_from_config():
    """Test building from config dictionary."""
    registry = ComponentRegistry()
    registry.register("fake_model", build_fake_model)

    config = {
        "provider": "fake_model",
        "model_name": "config-model",
        "temperature": 0.8,
    }

    model = registry.build_from_config(config)

    assert isinstance(model, FakeLanguageModel)
    assert model.model_name == "config-model"
    assert model.temperature == 0.8


def test_registry_build_from_config_missing_provider():
    """Test that build_from_config requires 'provider' key."""
    registry = ComponentRegistry()

    config = {"model_name": "test"}

    with pytest.raises(ValueError, match="must include 'provider'"):
        registry.build_from_config(config)


def test_registry_list_components():
    """Test listing registered components."""
    registry = ComponentRegistry()

    registry.register(
        "fake_model",
        build_fake_model,
        component_type="language_model",
        description="A fake model",
    )
    registry.register(
        "fake_retriever",
        build_fake_retriever,
        component_type="retriever",
        description="A fake retriever",
    )

    components = registry.list_components()

    assert len(components) == 2
    assert any(c["name"] == "fake_model" for c in components)
    assert any(c["name"] == "fake_retriever" for c in components)


def test_registry_list_components_filtered():
    """Test listing components filtered by type."""
    registry = ComponentRegistry()

    registry.register("fake_model", build_fake_model, component_type="language_model")
    registry.register(
        "fake_retriever", build_fake_retriever, component_type="retriever"
    )

    models = registry.list_components(component_type="language_model")

    assert len(models) == 1
    assert models[0]["name"] == "fake_model"


def test_registry_is_registered():
    """Test checking if component is registered."""
    registry = ComponentRegistry()

    registry.register("fake_model", build_fake_model)

    assert registry.is_registered("fake_model")
    assert not registry.is_registered("nonexistent")


def test_registry_unregister():
    """Test unregistering a component."""
    registry = ComponentRegistry()

    registry.register("fake_model", build_fake_model)
    assert registry.is_registered("fake_model")

    registry.unregister("fake_model")
    assert not registry.is_registered("fake_model")


def test_registry_clear():
    """Test clearing all components."""
    registry = ComponentRegistry()

    registry.register("fake_model", build_fake_model)
    registry.register("fake_retriever", build_fake_retriever)

    registry.clear()

    assert len(registry.list_components()) == 0


def test_global_registry():
    """Test that global registry is accessible."""
    registry = get_registry()
    assert isinstance(registry, ComponentRegistry)


def test_register_component_decorator():
    """Test the decorator for registering components."""
    # Create a fresh registry for this test
    test_registry = ComponentRegistry()

    # We'll manually register using the pattern the decorator uses
    @register_component("decorated_model", component_type="language_model")
    def build_decorated_model(**kwargs):
        return FakeLanguageModel(**kwargs)

    # The global registry should now have it
    global_registry = get_registry()
    assert global_registry.is_registered("decorated_model")

    # Build it to verify it works
    model = global_registry.build("decorated_model", model_name="decorated")
    assert isinstance(model, FakeLanguageModel)
    assert model.model_name == "decorated"


def test_registry_multiple_builds_create_separate_instances():
    """Test that building multiple times creates separate instances."""
    registry = ComponentRegistry()
    registry.register("fake_model", build_fake_model)

    model1 = registry.build("fake_model", model_name="model1")
    model2 = registry.build("fake_model", model_name="model2")

    assert model1 is not model2
    assert model1.model_name == "model1"
    assert model2.model_name == "model2"
