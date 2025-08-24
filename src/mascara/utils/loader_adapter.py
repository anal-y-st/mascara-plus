
from typing import Any, Mapping
def is_dataloader(obj: Any) -> bool:
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__") and hasattr(obj, "dataset")
def is_iterable_batches(obj: Any) -> bool:
    return hasattr(obj, "__iter__") and not hasattr(obj, "shape")
def _normalize_triplet(ret: Any):
    if isinstance(ret, Mapping):
        return ret.get("train"), ret.get("val"), ret.get("test")
    if isinstance(ret, tuple) and len(ret) in (1,2,3):
        if len(ret) == 1: return ret[0], None, None
        if len(ret) == 2: return ret[0], ret[1], None
        return ret
    return ret, None, None
def ensure_iterable(loader_like: Any):
    if loader_like is None: return None
    if is_dataloader(loader_like) or is_iterable_batches(loader_like): return loader_like
    raise TypeError("Unsupported loader type. Provide a torch DataLoader or an iterable yielding (x, y).")
def adapt_user_getters(obj: Any):
    train, val, test = _normalize_triplet(obj)
    return ensure_iterable(train), ensure_iterable(val), ensure_iterable(test)
