# ERA-v4 Study Repository

A structured workspace for a 26-week self-study program inspired by ERA V4. This repo includes a clean project layout, segmented requirements, dev tooling, and starter notebooks to help you progress from fundamentals to advanced topics.

## Quickstart (uv recommended)

Install `uv` (one-time):
- macOS (Homebrew): `brew install uv`
- Shell script: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Ensure `uv` is on PATH (if needed): `export PATH="$HOME/.local/bin:$PATH"`

Setup and run:
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
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
