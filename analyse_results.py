#!/usr/bin/env python
'''
analyse_results.py -- paired analysis of a results folder.

Every comparison is MATCHED on dataset and on all settings except the one
being varied, then tested with Wilcoxon signed-rank (efficiency) or an exact
binomial/McNemar test (validity). Marginal means across different dataset
subsets are not comparable, because coverage differs per configuration.

Usage:
  python analyse_results.py                         # results/
  python analyse_results.py results/2026_11_18      # a dated snapshot
  python analyse_results.py --eps 0.20              # one confidence level
'''
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    sys.exit('needs scipy: pip install scipy')

EPS_MAIN = 0.20          # confidence 0.80, the Figure 1 criterion


def load(root):
    files = sorted(glob.glob(os.path.join(root, 'metrics', '*', '*.csv')))
    if not files:
        sys.exit(f'no metrics files under {root}/metrics')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True, sort=False)
    print(f'{len(files)} files, {df.run_id.nunique()} runs, '
          f'{df.dataset.nunique()} datasets, {len(df)} rows\n')
    return df


def paired(df, col, va, vb, keys, label, metric='eff'):
    """Matched comparison of `col`==va against `col`==vb."""
    a = df[df[col] == va].set_index(keys)
    b = df[df[col] == vb].set_index(keys)
    a, b = a[~a.index.duplicated()], b[~b.index.duplicated()]
    j = a[[metric, 'valid']].join(b[[metric, 'valid']], how='inner',
                                  lsuffix='_a', rsuffix='_b')
    if len(j) < 6:
        print(f'{label:36s} only {len(j)} matched pairs -- skipped')
        return None
    d = j[f'{metric}_a'] - j[f'{metric}_b']
    try:
        p = stats.wilcoxon(j[f'{metric}_a'], j[f'{metric}_b']).pvalue
    except ValueError:                       # all differences zero
        p = 1.0
    # McNemar (exact) on the paired validity flags
    fixed = int((j.valid_a & ~j.valid_b).sum())
    broke = int((~j.valid_a & j.valid_b).sum())
    pv = stats.binomtest(fixed, fixed + broke, 0.5).pvalue if fixed + broke else np.nan
    print(f'{label:36s} n={len(j):3d}  {metric} {j[f"{metric}_a"].mean():.3f} vs '
          f'{j[f"{metric}_b"].mean():.3f} ({d.mean():+.3f}) p={p:.4f}  |  '
          f'valid {j.valid_a.mean():.0%} vs {j.valid_b.mean():.0%} '
          f'(fixed {fixed}, broke {broke}, p={pv:.4f})')
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', nargs='?', default='results')
    ap.add_argument('--eps', type=float, default=EPS_MAIN)
    args = ap.parse_args()

    df = load(args.root)
    c = df[np.isclose(df.eps, args.eps)].copy()
    print(f'--- comparisons at eps={args.eps:.2f} (confidence {1-args.eps:.2f}) ---\n')

    # ---- is the conformal calibration itself right? ----------------------
    print('CALIBRATION  mean(err_active) - eps, by calibration-set size')
    df['bin'] = pd.cut(df.n_active_train, [0, 100, 500, 10**9],
                       labels=['<=100', '101-500', '>500'])
    for eps in sorted(df.eps.unique()):
        cells = []
        for b in ['<=100', '101-500', '>500']:
            g = df[(df.bin == b) & np.isclose(df.eps, eps)]
            cells.append(f'{b} {g.err_active.mean()-eps:+.3f}' if len(g) else f'{b}   n/a')
        print('  eps=%.2f   %s' % (eps, '   '.join(cells)))
    print('  (negative = conservative. A strict err<=eps test is ~a coin flip\n'
          '   once calibration is exact, so judge validity with a binomial test.)\n')

    # ---- datasets that fail under every configuration --------------------
    bad = (c.groupby('dataset')
             .agg(runs=('run_id', 'nunique'), valid=('valid', 'mean'),
                  err_act=('err_active', 'mean'),
                  n_act_tr=('n_active_train', 'first'),
                  n_act_te=('n_active_test', 'first'))
             .query('valid < 0.1 and runs >= 5'))
    if len(bad):
        print('DATASETS INVALID UNDER EVERY CONFIGURATION (suspect the data, not the method)')
        print(bad.round(3).to_string(), '\n')

    K = ['dataset', 'features', 'dim', 'clf', 'tuned', 'balance']
    print('NODE FEATURES  18 (Table 1) vs 7 (original)')
    paired(c, 'node_features', 18, 7, K, 'nf18 vs nf7, all')
    for f in ['gvfa', 'concat', 'bundle']:
        paired(c[c.features == f], 'node_features', 18, 7, K, f'  within {f}')
    print()

    print('PROTOCOL')
    paired(c, 'balance', True, False,
           ['dataset', 'features', 'node_features', 'dim', 'clf', 'tuned'],
           'class balancing on vs off')
    paired(c, 'tuned', True, False,
           ['dataset', 'features', 'node_features', 'dim', 'clf', 'balance'],
           'Optuna tuned vs untuned (xgb)')
    paired(c, 'dim', 10000, 500,
           ['dataset', 'features', 'node_features', 'clf', 'tuned', 'balance'],
           'dim 10000 vs 500')
    print()

    print('FEATURE SETS')
    K2 = ['dataset', 'node_features', 'dim', 'clf', 'tuned', 'balance']
    for a, b in [('concat', 'gvfa'), ('concat', 'bundle'), ('gvfa', 'bundle'),
                 ('gvfa', 'rdkit'), ('concat', 'rdkit')]:
        paired(c, 'features', a, b, K2, f'{a} vs {b}')
    print()

    print('CLASSIFIERS')
    for a, b in [('xgb', 'rf'), ('xgb', 'lr'), ('rf', 'lr')]:
        paired(c, 'clf', a, b,
               ['dataset', 'features', 'node_features', 'dim', 'tuned', 'balance'],
               f'{a} vs {b}')
    print()

    print('COST  median seconds per run')
    print(c.groupby(['clf', 'tuned']).runtime_s.agg(['median', 'max', 'count'])
           .round(0).to_string())

    print('\nGRID COVERAGE  datasets present per configuration')
    print(c.groupby(['features', 'node_features', 'dim', 'clf', 'tuned', 'balance'])
           .dataset.nunique().to_string())


if __name__ == '__main__':
    main()
