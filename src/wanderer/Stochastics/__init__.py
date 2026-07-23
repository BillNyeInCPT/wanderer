# Exposes the entire stochastics module to the wanderer package. This allows users to access the StochasticProcess class directly from the wanderer package without needing to import it from the stochastics submodule.

# src/wanderer/Stochastics/__init__.py
import pkgutil
import importlib
import inspect

__all__ = []

for _, module_name, _ in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f".{module_name}", package=__name__)
    for name, obj in inspect.getmembers(module):
        if not name.startswith("_") and (inspect.isclass(obj) or inspect.isfunction(obj)):
            globals()[name] = obj
            __all__.append(name)