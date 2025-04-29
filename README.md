[![Build Status][build-img]][build-url]


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


[build-img]:https://github.com/pfrommerd/nanogen/workflows/pytest-uv/badge.svg
[build-url]:https://github.com/pfrommerd/nanogen/actions?query=workflow%3Apytest-uv
