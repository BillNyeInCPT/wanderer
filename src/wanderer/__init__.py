#Exposes the entire stochastics module to the wanderer package. This allows users to access the StochasticProcess class directly from the wanderer package without needing to import it from the stochastics submodule.

# src/wanderer/__init__.py
import pkgutil
import importlib

__all__ = []

for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f".{module_name}", package=__name__)
    if hasattr(module, "__all__"):
        for name in module.__all__:
            globals()[name] = getattr(module, name)
            __all__.append(name)