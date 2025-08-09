# ERA-v4 Study Repository

A structured workspace for a 26-week self-study program inspired by ERA V4. This repo includes a clean project layout, segmented requirements, dev tooling, and starter notebooks to help you progress from fundamentals to advanced topics.

## Quickstart (Recommended)

If your system Python venv is problematic, use the Virtualenv flow below.

### Option A: Virtualenv (most reliable)

```bash
python3 -m pip install --user virtualenv
python3 -m virtualenv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
pre-commit install
python -m ipykernel install --user --name era-v4 --display-name "Python (era-v4)"

jupyter lab
```

### Option B: Standard venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
pre-commit install
python -m ipykernel install --user --name era-v4 --display-name "Python (era-v4)"

jupyter lab
```

### Option C: uv (fast)

Install `uv` once: `curl -LsSf https://astral.sh/uv/install.sh | sh`

```bash
~/.local/bin/uv venv .venv
source .venv/bin/activate
~/.local/bin/uv pip install -r requirements.txt
pre-commit install
python -m ipykernel install --user --name era-v4 --display-name "Python (era-v4)"

jupyter lab
```

## Repo Layout

- `docs/` – course plan and notes
- `notebooks/` – weekly notebooks (start with `week01`)
- `src/` – reusable Python modules/utilities
- `requirements/` – segmented dependencies
- `scripts/` – helper CLI utilities (add as needed)
- `data/`, `models/`, `outputs/`, `logs/` – gitignored artifacts

## Notes on PyTorch

Install instructions vary by OS/GPU. If the default install fails, use the official selector: `https://pytorch.org/get-started/locally/`. For GPU training, Colab/Kaggle is often the quickest path.

## License

MIT
