# Running the whole experiment set on NCI Gadi

Everything below is run from the project root on Gadi:

```
/scratch/jq77/nk8155/Binary_classification/binary_classification
```

Every job is **resumable**: `nci/make_commands.py` skips any run whose metrics
file already exists, and any cache/split already built. If a job hits its
walltime, submit exactly the same command again and it continues where it
stopped.

---

## What "all the experiments" is

| | count |
|---|---|
| datasets | **45** (23 ED / `msc_new`, 12 `tox21.data`, 10 `HTSFP10`) |
| feature sets | 4 — `gvfa`, `concat`, `bundle`, `rdkit` |
| node features | 2 — 7 (original) and 18 (Table 1) |
| dimensions | 6 — 100, 500, 1000, 2000, 5000, 10000 |
| classifiers | 3 — `xgb`, `rf`, `lr` |
| configs per dataset | 3x2x6 + 1 = 37 feature configs x 3 classifiers = **111** |
| **total runs** | **4,995** |

Plus the one-off caches: 45 prepare tasks (descriptors + cross-conformal
split) and **540** embedding tasks (45 x 2 node-features x 6 dims).

Two passes over the grid:

* `TUNE=0` — all 4,995 runs with default classifier settings.
* `TUNE=50` — adds only the 1,665 **xgb** runs with 50 Optuna trials.
  `--tune` acts on xgb alone and leaves the `rf`/`lr` run tags unchanged, so
  the second pass skips everything the first pass already did.

### Before you start: budget

* **Disk.** The embedding cache for all 45 datasets at all 6 dimensions and
  both node-feature settings is about **110 GB** (783k molecules x 37,200
  floats). Check your room with `lquota`. If it is tight, run one dimension
  at a time — Step 4b.
* **Service units.** One 48-core 48-hour job is roughly **4,600 SU**. The
  tuned xgb pass is the expensive part. Run the pilot in Step 2 first and use
  its real timings before committing to the full grid.

---

## Step 0 — push the fixed harness to Gadi

Job 184162620 lost all 551 of its tasks because `nci-parallel` execs its input
lines **without a shell**, so the old `cd ... && VAR=x python ... > log` lines
could never run. `nci/run_task.sh` now does the `cd`, the thread limits and
the redirection, and `make_commands.py` emits plain word lists. Copy the three
files up before resubmitting anything:

```
rsync -av nci/run_task.sh nci/make_commands.py nci/sweep.pbs nci/RUN_ALL.md gadi-dm.nci.org.au:/scratch/jq77/nk8155/Binary_classification/binary_classification/nci/
```

---

## Step 1 — one-off checks (login node, no job)

Which project holds your compute, and how much scratch you have:

```
nci_account
```

```
lquota
```

`nci/sweep.pbs` now says `#PBS -P mi23` with `#PBS -l storage=scratch/jq77`,
because both earlier jobs reported `Project: mi23` while the code and data sit
on `/scratch/jq77`. That split is normal: compute charged to one project,
storage mounted from another. Only change it if `nci_account` disagrees —
and then change both lines together.

Now check the Python the jobs will use — the two paths the script defaults to:

```
ls -d /scratch/jq77/nk8155/Binary_classification/binary_classification && ls -l /scratch/jq77/nk8155/tft_env/bin/python
```

`sweep.pbs` defaults to exactly those, so no `-v ROOT=` or `-v PY=` is needed:

```
module load python3/3.8.5 && . /scratch/jq77/nk8155/tft_env/bin/activate && python -c "import sys,numpy,pandas,sklearn,torch,xgboost,rdkit,torch_geometric,optuna; print(sys.version); print('all imports ok')"
```

If `optuna` is the only thing missing, install it into that venv (needs
internet, so a login node or the `copyq` queue):

```
pip install --no-cache-dir optuna
```

Finally, confirm the command generator sees all 45 datasets and counts the
grid correctly:

```
python nci/make_commands.py --groups ED tox21 HTSFP10 --tune 50 --out-dir /tmp/cmdcheck
```

Expect `45 commands`, `540 commands`, and `experiments: 4995 in the grid`.

Look at one generated line. It must be plain words — no `cd`, no `&&`, no
`>` — starting with `/bin/bash .../nci/run_task.sh`:

```
head -1 /tmp/cmdcheck/stage1_prepare.txt
```

You can run that one line by hand to smoke-test the whole chain before
queuing anything; it builds one dataset's descriptors and split:

```
eval "$(head -1 /tmp/cmdcheck/stage1_prepare.txt)" ; echo "exit=$?" ; tail -3 logs/ED/*__prepare.log
```

> If you build the pinned `gvfa` conda environment later, submit with
> `-v PY=/scratch/jq77/nk8155/envs/gvfa/bin/python,PY_MODULES=`
> — an empty `PY_MODULES`, because a conda env needs no module loaded.

---

## Step 2 — pilot, to measure run times (4 hours)

One group, no tuning, one dimension. It builds its own caches, so it also
proves stages 1 and 2 work before you commit a 48-hour job.

```
qsub -v SWEEP_GROUPS=ED,TUNE=0,DIMS=500,NODE_FEATURES=18 -l walltime=04:00:00 nci/sweep.pbs
```

When it finishes:

```
ls results/metrics/ED | wc -l
```

Every metrics file carries a `runtime_s` column, so merge them and read the
real per-run cost straight off it:

```
python collect_results.py --latest && python -c "import pandas as pd; d=pd.read_csv('results/all_results.csv').drop_duplicates('run_id'); print(d['runtime_s'].describe()); print(d.nlargest(8,'runtime_s')[['tag','runtime_s']].to_string(index=False))"
```

Multiply the slowest run by 4,995/4 tasks to sanity-check the full grid, and
by 1,665/4 for the tuned pass. **Do not skip this step** — it is what tells
you whether to trim the grid (last section).

---

## Step 3 — prepare and embed, once per group (stages 1 + 2)

These write the caches every run reads. One job per group; the three can run
at the same time. Never run two stage-1 jobs for the same group at once.

```
qsub -v SWEEP_GROUPS=ED,STAGES=1+2 nci/sweep.pbs
```

```
qsub -v SWEEP_GROUPS=tox21,STAGES=1+2 nci/sweep.pbs
```

```
qsub -v SWEEP_GROUPS=HTSFP10,STAGES=1+2 nci/sweep.pbs
```

`HTSFP10` is by far the biggest (569k molecules) and will probably need
resubmitting once or twice. Progress:

```
ls cache/*_dim10000.npy | wc -l
```

(180 when every dim-10000 embedding exists.)

```
du -sh cache
```

---

## Step 4 — the untuned grid (stage 3, `TUNE=0`)

4,995 runs. Split each group across three nodes by classifier.

### 4a — one group at a time, three nodes

```
qsub -v SWEEP_GROUPS=ED,STAGES=3,TUNE=0,CLFS=xgb nci/sweep.pbs
```

```
qsub -v SWEEP_GROUPS=ED,STAGES=3,TUNE=0,CLFS=rf nci/sweep.pbs
```

```
qsub -v SWEEP_GROUPS=ED,STAGES=3,TUNE=0,CLFS=lr nci/sweep.pbs
```

Then the same three lines with `SWEEP_GROUPS=tox21`, and again with
`SWEEP_GROUPS=HTSFP10`. Resubmit any line whenever its job runs out of
walltime.

### 4b — if disk is tight, one dimension at a time

Embed and run a single dimension, then delete that dimension's cache before
moving on:

```
qsub -v SWEEP_GROUPS=HTSFP10,STAGES=2+3,TUNE=0,DIMS=10000 nci/sweep.pbs
```

and once it reports every run finished:

```
rm -f cache/*_dim10000.npy
```

---

## Step 5 — the tuned xgb pass (stage 3, `TUNE=50`)

1,665 runs, 50 Optuna trials each: the expensive pass. Only `xgb` is
affected, so restrict it to that classifier.

```
qsub -v SWEEP_GROUPS=ED,STAGES=3,TUNE=50,CLFS=xgb nci/sweep.pbs
```

```
qsub -v SWEEP_GROUPS=tox21,STAGES=3,TUNE=50,CLFS=xgb nci/sweep.pbs
```

```
qsub -v SWEEP_GROUPS=HTSFP10,STAGES=3,TUNE=50,CLFS=xgb nci/sweep.pbs
```

To spread one group over more nodes, give each job part of the dimension
grid — two jobs on disjoint parts never write the same file:

```
qsub -v SWEEP_GROUPS=HTSFP10,STAGES=3,TUNE=50,CLFS=xgb,DIMS=100+500+1000 nci/sweep.pbs
```

```
qsub -v SWEEP_GROUPS=HTSFP10,STAGES=3,TUNE=50,CLFS=xgb,DIMS=2000+5000+10000 nci/sweep.pbs
```

---

## Step 6 — collect the results

Every job runs this at the end, but you can run it any time on a login node:

```
module load python3/3.8.5 && . /scratch/jq77/nk8155/tft_env/bin/activate && python collect_results.py --latest
```

It writes `results/all_results.csv` and prints the confidence-0.80 summary:
valid models and mean singleton rate per configuration.

How many of the 4,995 are done:

```
find results/metrics -name '*.csv' | wc -l
```

Copy the results to your laptop — metrics and configs only, not the large
per-molecule p-value files:

```
rsync -av --include='*/' --include='metrics/***' --include='configs/***' --include='all_results.csv' --exclude='*' gadi-dm.nci.org.au:/scratch/jq77/nk8155/Binary_classification/binary_classification/results/ ./results/
```

---

## Monitoring and troubleshooting

```
qstat -u $USER
```

```
qstat -swx <jobid>
```

```
tail -f gvfa_sweep.o<jobid>
```

```
grep -l Traceback logs/*/*.log | head -20
```

Each stage now aborts the job if it launches tasks and **no** log file
appears, printing the first task line — that is the signature of tasks that
cannot start at all, as in job 184162620. If you see it, run that printed
line by hand on a login node; it is a plain command and will show you the
real error.

A task killed for memory or for the timeout leaves a log with no `saved`
line; the end of every job lists those. If tasks are being killed, give each
one more memory by running fewer side by side:

```
qsub -v SWEEP_GROUPS=HTSFP10,STAGES=3,TUNE=0,NCORES_PER_TASK=24 nci/sweep.pbs
```

`NCORES_PER_TASK=12` is 4 tasks x ~47 GB (the default), `24` is 2 x ~95 GB,
`48` is one task with the whole 190 GB.

Cancel a job, and anything waiting on it:

```
qdel <jobid>
```

### Passing lists through `qsub -v`

`qsub -v` separates variables with commas, so a **list uses plus signs**:

```
-v DIMS=500+1000,CLFS=rf+lr,STAGES=1+2
```

Never put a space inside `-v`.

---

## Trimming the grid, if the pilot says the full run is too expensive

In order of what costs most for what it buys:

1. Drop `TUNE=50` everywhere except the one group the paper tunes on, and
   report the untuned grid for the rest.
2. Cut the dimensions to `DIMS=500+2000+10000` — halves both the embedding
   cache and the number of runs.
3. Run `NODE_FEATURES=18` only, keeping 7 as a comparison on one group:
   `-v NODE_FEATURES=7,SWEEP_GROUPS=ED`.
4. Drop `bundle` (`FEATURES=gvfa+concat`). `concat` was the best in the
   draft, and `bundle` has never been run end to end.
