#!/bin/bash
# Build the regularizedvi conda env on Crick.
#
# Usage:
#   # On an interactive GPU node (so the CUDA smoke test can run):
#   bash /camp/home/kleshcv/srun.sh -p vis -- bash scripts/helper_scripts/build_conda_env_crick.sh
#
#   # Or on a login node (imports verified; torch.cuda.is_available() will be False):
#   bash scripts/helper_scripts/build_conda_env_crick.sh
#
# Requires scripts/helper_scripts/setup_bashrc_crick.sh to have been run once.

set -euo pipefail

ENV_PREFIX="/nemo/lab/briscoej/home/users/kleshcv/conda_environments/regularizedvi"
REPO_ROOT="/nemo/lab/briscoej/home/users/kleshcv/my_packages/regularizedvi"

echo "=== Step 1: Load module + conda via ~/.bashrc ==="
# Disable -u around source: /etc/bashrc references unset $BASHRCSOURCED.
set +u
# shellcheck source=/dev/null
source "$HOME/.bashrc"
set -u

if ! type conda >/dev/null 2>&1 || ! type conda | head -1 | grep -q "function"; then
    echo "ERROR: conda function not available after sourcing ~/.bashrc." >&2
    echo "Run: bash scripts/helper_scripts/setup_bashrc_crick.sh  to check prerequisites." >&2
    exit 1
fi

echo "=== Step 2: Create env at $ENV_PREFIX (python=3.11) ==="
export PYTHONNOUSERSITE=TRUE
if [ -d "$ENV_PREFIX" ]; then
    echo "Env already exists — skipping create. Delete it first to rebuild from scratch."
else
    conda create -y -p "$ENV_PREFIX" python=3.11
fi

echo "=== Step 3: Activate (PYTHONNOUSERSITE=TRUE explicit) ==="
export PYTHONNOUSERSITE=TRUE
# Redirect pip cache onto lab NEMO (5 GB homefs quota would overflow otherwise).
export PIP_CACHE_DIR=/nemo/lab/briscoej/home/users/kleshcv/.cache/pip
mkdir -p "$PIP_CACHE_DIR"
conda activate "$ENV_PREFIX"
echo "  CONDA_PREFIX=$CONDA_PREFIX"
echo "  which python: $(which python)"
echo "  PYTHONNOUSERSITE=$PYTHONNOUSERSITE"
echo "  PIP_CACHE_DIR=$PIP_CACHE_DIR"

echo "=== Step 4: Install torch (GPU, cu124) ==="
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "=== Step 5: Install regularizedvi + test deps (mirrors GitHub CI hatch-test env) ==="
pip install -e "${REPO_ROOT}[test]"

echo "=== Step 6: Install user-facing packages present in Sanger env ==="
pip install \
    jupyter jupyterlab ipykernel ipython notebook nbconvert nbformat \
    matplotlib seaborn plottable \
    pandas pyarrow openpyxl h5py \
    numba pynndescent umap-learn \
    scikit-learn statsmodels \
    scib-metrics \
    papermill wandb session-info2 \
    pre-commit cruft \
    loompy rds2py biocutils \
    ml_collections pydantic \
    tqdm rich

echo "=== Step 7: Install gh CLI + nodejs (conda-forge) ==="
# nodejs is needed so pre-commit's biome hook can use the in-env node
# (language_version: system in .pre-commit-config.yaml). The node binary
# pre-commit otherwise downloads links against libatomic.so.1, which is
# absent on RHEL8/Rocky 8 by default.
conda install -y -c conda-forge gh nodejs

echo "=== Step 8: Smoke tests ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import regularizedvi, scvi, scanpy, mudata, anndata, papermill, wandb, pyro, lightning; print('imports ok')"
gh --version

echo ""
echo "=== Build complete ==="
echo "Env: $ENV_PREFIX"
echo "Activate from any shell with:   conda activate $ENV_PREFIX"
echo "Or use the launcher:            bash scripts/helper_scripts/run_python_cmd.sh <script>"
