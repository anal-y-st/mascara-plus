import importlib
import importlib.util
import os
import sys

def load_dataloader_module(module_path: str):
    """Charge un module Python soit depuis un chemin .py, soit depuis un nom de module."""
    if module_path.endswith(".py") and os.path.isfile(module_path):
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod
    else:
        return importlib.import_module(module_path)
