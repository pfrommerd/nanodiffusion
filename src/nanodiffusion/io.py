import safetensors
import tempfile
import torch
import numpy as np
import json
import typing as tp
import os

from pathlib import Path

################ IO Utilities ######################

def _flatten_dict(data, prefix=""):
    for k, v in data.items():
        if isinstance(v, dict):
            yield from _flatten_dict(v, f"{prefix}{k}::")
        else:
            yield (f"{prefix}{k}", v)

def _unflatten_dict(data, target=None):
    out = target if target is not None else {}
    for k, v in data.items():
        keys = k.split("::")
        d = out
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = v
    return out

def save(file, data : dict[str, tp.Any]):
    data = dict(_flatten_dict(data))
    # Get all non-tensor objects
    metadata = {k:v for k,v in data.items() if not isinstance(v, (torch.Tensor, np.ndarray))}
    metadata = _unflatten_dict(metadata)
    metadata = {k: json.dumps(v) for k,v in metadata.items()}
    # Get all tensor objects
    tensors = {k:v for k,v in data.items() if isinstance(v, (torch.Tensor, np.ndarray))}
    tensors = safetensors.torch._flatten(tensors)
    if isinstance(file, (str, Path)):
        safetensors.serialize_file(tensors, file, metadata)
    else:
        data = safetensors.serialize(tensors, metadata)
        file.write(data)

def load(file, device=None) -> dict[str, tp.Any]:
    metadata = {}
    tensors = {}
    temp_path = None
    try:
        if not isinstance(file, (str, Path)):
            _, temp_path = tempfile.mkstemp(suffix=".safetensors")
            with open(temp_path, "wb") as f:
                if isinstance(file, bytes):
                    f.write(file)
                else:
                    f.write(file.read())
            file = temp_path
        with safetensors.safe_open(file, framework="pt", device=device) as f:
            # Read in the metadata
            metadata = dict(f.metadata())
            metadata = {k: json.loads(v) for k, v in metadata.items()}
            # Read in the tensors
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    finally:
        if temp_path:
            os.remove(temp_path)
    return _unflatten_dict(tensors, target=metadata)