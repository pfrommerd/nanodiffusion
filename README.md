[![Build Status][build-img]][build-url]

# Code for experiments for diffusion with anisotropic noise


### To install, first install UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then install the dependencies (including development dependencies)
```bash
uv sync --dev
```

### To train, run

```bash
uv run train
```


[build-img]:https://github.com/pfrommerd/nanodiffusion/workflows/pytest-uv/badge.svg
[build-url]:https://github.com/pfrommerd/nanodiffusion/actions?query=workflow%3Apytest-uv
