#!/bin/bash
# =============================================================================
# make_env.sh -- build the pinned `gvfa` environment on NCI Gadi, entirely on
# /scratch so nothing lands in your 10 GB home quota.
#
# Why: tft_env is a Python 3.11 TensorFlow venv. It is missing
# typing_extensions (torch needs it), its pip is broken
# ("No module named 'pip._internal.cli.main'"), and environment.yml is the
# paper's own export (prefix: /home/ulfno/anaconda3a/envs/gvfa) pinning
# python 3.8.20, numpy 1.24.4, rdkit 2023.09.5, scikit-learn 1.0.2,
# torch 2.0.1, torch-geometric 2.6.1, xgboost 1.6.2 -- so numbers from
# tft_env are not comparable to the published ones.
#
# Run on a LOGIN NODE (it needs internet; compute nodes have none), from a
# CLEAN shell -- no venv active, no python module loaded:
#   deactivate 2>/dev/null; module purge; bash nci/make_env.sh
#
# Safe to re-run: it skips Miniforge and the env if they already exist.
# Afterwards, submit with no extra flags -- sweep.pbs picks the env up via:
#   qsub -v PY=/scratch/jq77/nk8155/envs/gvfa/bin/python nci/sweep.pbs
# (PY_MODULES is already empty by default, which is right for a conda env.)
# =============================================================================
set -euo pipefail

# ---- refuse to run in a polluted shell -----------------------------------
# tft_env broke exactly this way: PYTHONPATH/PYTHONHOME left over from a
# `module load python3/...` made a 3.11.0 interpreter load 3.11.7's compiled
# stdlib, giving "undefined symbol: _PyModule_Add" on every C extension. The
# same leak would poison the env this script builds, so clear it and stop if a
# venv or python module is still active.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONNOUSERSITE
export PYTHONNOUSERSITE=1          # never pick up ~/.local/lib packages

if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "ERROR: the venv at $VIRTUAL_ENV is active. Run 'deactivate' first."
    exit 1
fi
if [ -n "${LOADEDMODULES:-}" ] && echo "$LOADEDMODULES" | grep -qi python; then
    echo "ERROR: a python module is loaded ($LOADEDMODULES)."
    echo "Run 'module purge' first, then re-run this script."
    exit 1
fi

S=${S:-/scratch/jq77/nk8155}
ENV_PREFIX=${ENV_PREFIX:-$S/envs/gvfa}
FORGE=$S/miniforge3
YML=${YML:-$(cd "$(dirname "$0")/.." && pwd)/environment.yml}

# ---- keep every cache off $HOME ------------------------------------------
export CONDA_PKGS_DIRS=$S/conda_pkgs
export PIP_CACHE_DIR=$S/cache/pip
export XDG_CACHE_HOME=$S/cache
export TMPDIR=$S/tmp
export CONDA_ENVS_PATH=$S/envs
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR" "$S/envs"

echo "== caches redirected to $S (home quota untouched)"
echo "   env will be: $ENV_PREFIX"
echo "   recipe:      $YML"
[ -f "$YML" ] || { echo "ERROR: no environment.yml at $YML"; exit 1; }

# ---- 1. Miniforge --------------------------------------------------------
if [ -x "$FORGE/bin/conda" ]; then
    echo "== Miniforge already at $FORGE"
else
    echo "== downloading Miniforge to $S"
    cd "$S"
    URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    curl -fL --retry 3 -o miniforge.sh "$URL" || {
        echo
        echo "ERROR: download failed. Login nodes reach the internet through a"
        echo "proxy; if it is unavailable, run this script in the copyq queue"
        echo "instead, or fetch Miniforge3-Linux-x86_64.sh on your laptop and"
        echo "scp it to $S/miniforge.sh, then re-run."
        exit 1
    }
    bash miniforge.sh -b -p "$FORGE"
    rm -f miniforge.sh
fi
export PATH=$FORGE/bin:$PATH

# ---- 2. the environment --------------------------------------------------
if [ -x "$ENV_PREFIX/bin/python" ]; then
    echo "== env already at $ENV_PREFIX (delete it to rebuild)"
else
    echo "== solving environment.yml -- this takes 10-30 minutes, be patient"
    # The yml is a full export with exact builds and several narrow channels,
    # so the solve can fail. If it does, the message below says what to send.
    conda env create -f "$YML" -p "$ENV_PREFIX" || {
        echo
        echo "ERROR: the pinned solve failed -- most likely one exact conda"
        echo "build in the yml is no longer on the channel. Fallback that keeps"
        echo "every version the paper used, without the exact build strings:"
        echo
        echo "  conda create -y -p $ENV_PREFIX -c conda-forge python=3.8.20 \\"
        echo "      numpy=1.24.4 rdkit=2023.09.5 pandas scipy"
        echo "  $ENV_PREFIX/bin/pip install --no-cache-dir \\"
        echo "      scikit-learn==1.0.2 torch==2.0.1 torch-geometric==2.6.1 \\"
        echo "      xgboost==1.6.2 scikit-posthocs==0.8.1 optuna"
        echo
        echo "Send me the last ~20 lines above if you would rather I adjust it."
        exit 1
    }
fi

# optuna is NOT in environment.yml (the paper did no hyper-parameter search),
# but --tune needs it. Added separately so the pinned set stays untouched.
if "$ENV_PREFIX/bin/python" -c 'import optuna' 2>/dev/null; then
    echo "== optuna already present"
else
    echo "== adding optuna (needed by --tune; not part of the pinned set)"
    "$ENV_PREFIX/bin/pip" install --no-cache-dir optuna || {
        echo "WARNING: optuna not installed. Untuned runs (TUNE=0) still work."
    }
fi

# ---- 3. prove it ---------------------------------------------------------
echo "== checking every import the runs need"
"$ENV_PREFIX/bin/python" -c "
import sys
print('python', sys.version.split()[0], sys.executable)
bad = []
for m in ['typing_extensions','numpy','scipy','pandas','sklearn','joblib',
          'torch','torch_geometric','xgboost','rdkit','optuna']:
    try:
        __import__(m); print('  OK   ', m)
    except Exception as e:
        bad.append(m); print('  FAIL ', m, type(e).__name__, e)
sys.exit(1 if bad else 0)
" || {
    echo
    echo "Some imports failed. Install just those into the env, e.g.:"
    echo "  $FORGE/bin/conda install -p $ENV_PREFIX -c conda-forge optuna"
    exit 1
}

cat <<EOF

== done. Use it with:

  qsub -v PY=$ENV_PREFIX/bin/python,SWEEP_GROUPS=ED,TUNE=0,DIMS=500 -l walltime=04:00:00 nci/sweep.pbs

To make it the default and stop passing PY=, edit the PY line in nci/sweep.pbs:

  PY=\${PY:-$ENV_PREFIX/bin/python}
EOF
