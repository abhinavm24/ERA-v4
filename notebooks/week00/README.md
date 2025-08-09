# Week 0

Start with:
- `00_mac_torch_gpu_acceleration.ipynb` – macOS MPS quick check. Installs `requirements/cv.txt` via `uv`, asserts MPS availability, and runs a small matmul on the `mps` device. Recommended for Apple Silicon.
- `01_mlp_xor.ipynb` – implement an MLP from scratch using NumPy
- `02_pytorch_mlp.ipynb` – replicate with PyTorch and compare
    If you skipped the MPS check notebook, install deps first:
    - `uv pip install -r requirements/cv.txt`

Tip: Use Colab if you prefer a hosted environment. You can upload this notebook folder directly or clone the repo in Colab.
