# results/

Written by `Run_experiment.py`. One set of three files per run; nothing is
appended, so parallel PBS jobs never write to the same file.

```
results/
  metrics/<group>/<run_id>.csv            4 rows: confidence 0.70 / 0.80 / 0.90 / 0.95
  pvalues/<group>/<run_id>_p-values.csv   per test molecule: id, p(0), p(1), class
  configs/<group>/<run_id>.json           all arguments, classifier parameters,
                                          tuning result, package versions, host, PBS job
  all_results.csv                         made by collect_results.py
```

`<group>` is the `--group` argument (`ED`, `tox21`, `HTSFP10`), or `nogroup`.

## File names

```
<dataset>_<features>[_nf<N>_dim<D>]_<clf>[_bal][_tune<T>]_seed<S>_k<K>__<YYYYmmdd-HHMMSS>[_job<PBS id>]
```

| part | meaning |
|---|---|
| `dataset` | `AID_743122`, `tox21_NR-AhR`, `HTSFP10_1814` |
| `features` | `gvfa`, `concat`, `bundle`, `rdkit` |
| `nf<N>` | node features, 7 or 18 (not for `rdkit`) |
| `dim<D>` | hypervector dimension (not for `rdkit`) |
| `clf` | `xgb`, `rf`, `lr` |
| `bal` | class balancing on (`--balance`) |
| `tune<T>` | Optuna with T trials (xgb only) |
| `seed<S>`, `k<K>` | split seed, number of cross-conformal folds |
| time stamp | when the run started; `_job…` added inside a PBS job |

Example: `AID_743122_gvfa_nf18_dim500_xgb_seed42_k10__20261109-185214_job123456.csv`

The part before `__` is also stored as the `tag` column, so repeats of the same
setting share a `tag` and differ in `run_id`.

## Analysis

```bash
python collect_results.py            # every run -> results/all_results.csv
python collect_results.py --latest   # newest run of each setting only
```

Each metrics row also carries `clf_params` (JSON) and the dataset sizes, so the
merged table is enough for Figures 1-3 without opening the config files.
