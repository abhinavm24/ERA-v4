### Development setup

1. Create a virtual environment and install dev deps:
   ```bash
   uv venv .venv && source .venv/bin/activate
   uv pip install -r requirements.txt
   pre-commit install
   ```

2. Run checks locally:
   ```bash
   make fmt lint mypy nbqa test
   ```

3. Clean notebook outputs before committing (pre-commit hook will also do this):
   ```bash
   bash scripts/clean_notebooks.sh
   ```

### Adding notebooks

- Place new notebooks under `notebooks/weekNN/`.
- Keep outputs light; rely on `nbstripout` to clear heavy outputs in commits.
- If notebooks require extra deps, add them to a new or existing file in `requirements/`.
