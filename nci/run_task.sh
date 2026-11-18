#!/bin/bash
# =============================================================================
# run_task.sh -- run ONE experiment task under nci-parallel.
#
# nci-parallel does NOT pass its input lines through a shell: it splits each
# line into words and execs the first one. So a task line cannot contain
#   cd ... &&          (job 184162620: "/bin/cd: line 2: cd: too many arguments"
#   VAR=value prefixes  551 times, one per task, every task dead in 2 seconds)
#   > logfile 2>&1
#
# This script does all of that itself, so each task line is a plain list of
# whitespace-separated words that execs cleanly with no shell involved.
#
# Called by nci/make_commands.py as:
#   /bin/bash <this script> <root> <threads> <logfile> <python> <args for Run_experiment.py...>
# =============================================================================
set -u

if [ "$#" -lt 5 ]; then
    echo "usage: run_task.sh <root> <threads> <logfile> <python> <args...>" >&2
    exit 2
fi

ROOT=$1
THREADS=$2
LOG=$3
PY=$4
shift 4

cd "$ROOT" || { echo "run_task.sh: cannot cd to $ROOT" >&2; exit 1; }

# A venv built on a module python cannot start without that module's
# LD_LIBRARY_PATH. Report that in the JOB log, rather than leaving thousands of
# task logs each holding one cryptic linker error.
if ! "$PY" -c '' 2>/dev/null; then
    echo "run_task.sh: $PY cannot start. Is PY_MODULES right in sweep.pbs?" >&2
    echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<empty>}" >&2
    "$PY" -c '' || true          # let the real linker error through to the log
    exit 1
fi
mkdir -p "$(dirname "$LOG")" || exit 1

# one task must not grab the whole node's cores: each gets THREADS of them
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export TF_CPP_MIN_LOG_LEVEL=3
# PYTHONUNBUFFERED: a task killed by the OOM killer or the timeout still leaves
# its log up to the point it died, instead of an empty file.
export PYTHONUNBUFFERED=1

exec "$PY" Run_experiment.py "$@" > "$LOG" 2>&1
