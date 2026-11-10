#!/usr/bin/env python
'''
collect_results.py -- merge every per-run metrics file into one table.

Run_experiment.py writes one file per run to results/metrics/<group>/<run_id>.csv
(4 rows, one per confidence level). This script concatenates them so Figures
1-3 are a groupby over one DataFrame.

Usage:
  python collect_results.py                      # -> results/all_results.csv
  python collect_results.py --latest             # newest run per setting only
  python collect_results.py --group ED --out results/ED.csv

--latest keeps, for each `tag` (dataset + all settings, without the time
stamp), only the most recent run. Use it after re-running a setting; leave it
off to keep every repeat (e.g. to look at run-to-run variation).
'''
import argparse
import os
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--group', default=None, help='only this group sub-folder')
    ap.add_argument('--latest', action='store_true',
                    help='keep only the newest run of each setting')
    ap.add_argument('--out', default=None,
                    help='default: <results-dir>/all_results.csv')
    args = ap.parse_args()

    base = Path(args.results_dir) / 'metrics'
    if args.group:
        base = base / args.group
    files = sorted(base.rglob('*.csv'))
    if not files:
        raise SystemExit(f'no metrics files under {base}')

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True, sort=False)
    n_runs = df['run_id'].nunique()

    if args.latest:
        newest = (df.sort_values('started')
                    .drop_duplicates('tag', keep='last')['run_id'])
        df = df[df['run_id'].isin(newest)]

    out = Path(args.out) if args.out else Path(args.results_dir) / 'all_results.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f'{out.name}.{os.getpid()}.tmp')   # two jobs finishing at
    df.to_csv(tmp, index=False)                            # once never leave a
    os.replace(tmp, out)                                   # half-written table

    print(f'{len(files)} files, {n_runs} runs -> {df["run_id"].nunique()} runs '
          f'kept, {len(df)} rows written to {out}')

    # quick overview: valid models at confidence 0.8 (the Figure 1 criterion)
    c8 = df[df['conf'].round(2) == 0.8]
    if len(c8):
        keys = ['features', 'node_features', 'dim', 'clf', 'tuned', 'balance']
        summary = (c8.groupby(keys, dropna=False)
                     .agg(runs=('run_id', 'nunique'),
                          valid=('valid', 'sum'),
                          mean_eff=('eff', 'mean'))
                     .sort_values('valid', ascending=False))
        print('\nconfidence 0.80 -- valid models and mean singleton rate:')
        print(summary.to_string())


if __name__ == '__main__':
    main()
