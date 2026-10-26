"""Circuit modules and arithmetic implementations, imported on demand."""

from importlib import import_module

__all__ = ["FixedPointArithmetic", "RotationBoltzmannCoin", "HybridPhaseArithmetic", "fixed_point_arithmetic", "hybrid_arithmetic"]


def __getattr__(name):
    modules = {
        "FixedPointArithmetic": "arithmetic_fixed_point",
        "RotationBoltzmannCoin": "arithmetic_fixed_point",
        "HybridPhaseArithmetic": "arithmetic_hybrid",
        "fixed_point_arithmetic": "arithmetic_fixed_point",
        "hybrid_arithmetic": "arithmetic_hybrid",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{modules[name]}")
    value = module if name.endswith("_arithmetic") else getattr(module, name)
    globals()[name] = value
    return value
