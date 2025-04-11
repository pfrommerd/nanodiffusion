def flatten_items(data, prefix=""):
    for k, v in data.items():
        if prefix:
            k = f"{prefix}.{k}"
        if isinstance(v, dict):
            yield from flatten_items(v, k)
        else:
            yield (k, v)

def flatten_dict(data, prefix=""):
    return dict(flatten_items(data, prefix=prefix))