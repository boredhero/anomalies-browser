"""Detection pass registry with auto-registration decorator."""

from typing import Type

from magic_eyes.detection.base import DetectionPass


class PassRegistry:
    """Singleton registry for detection passes."""

    _instance: "PassRegistry | None" = None
    _passes: dict[str, Type[DetectionPass]] = {}

    def __new__(cls) -> "PassRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, pass_class: Type[DetectionPass]) -> Type[DetectionPass]:
        """Register a detection pass class."""
        instance = pass_class()
        cls._passes[instance.name] = pass_class
        return pass_class

    @classmethod
    def get(cls, name: str) -> Type[DetectionPass]:
        """Get a registered pass class by name."""
        if name not in cls._passes:
            available = list(cls._passes.keys())
            raise KeyError(f"Unknown pass: {name!r}. Available: {available}")
        return cls._passes[name]

    @classmethod
    def list_passes(cls) -> dict[str, Type[DetectionPass]]:
        """Return all registered passes."""
        return dict(cls._passes)

    @classmethod
    def get_pass_chain(cls, names: list[str]) -> list[DetectionPass]:
        """Instantiate a chain of passes by name."""
        return [cls.get(name)() for name in names]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered passes. Useful for testing."""
        cls._passes.clear()


def register_pass(cls: Type[DetectionPass]) -> Type[DetectionPass]:
    """Decorator to register a detection pass at import time."""
    return PassRegistry.register(cls)
