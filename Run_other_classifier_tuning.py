#!/usr/bin/env python
"""Tune Random Forest or Logistic Regression with conformal efficiency.

This is the RF/LR companion to Run_experiment.py. It reuses the same cached
features, shared splits, cross-conformal prediction, output format and tuning
objective. Optuna searches on training data only and maximises singleton
efficiency at 80% confidence; invalid candidates receive a one-point penalty.
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from Run_experiment import (
    EPS_LEVELS,
    assemble,
    classifier_params,
    cp_metrics,
    cross_conformal_pvalues,
    gvfa_embeddings,
    new_run_id,
    package_versions,
    rdkit_descriptors,
)
from src.load_data import (
    check_feasible,
    dataset_name_from_path,
    fold_indices,
    get_splits,
    labels_of,
    load_full,
    min_calib_per_class,
)
from src.result_naming import run_tag, short_dataset_name


def rf_search_params(trial):
    """Random Forest search space, including the class-weight strategy."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
        'max_depth': trial.suggest_categorical(
            'max_depth', [None, 8, 16, 24, 32]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical(
            'max_features', ['sqrt', 'log2', 0.5, 0.8]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': trial.suggest_categorical(
            'class_weight', [None, 'balanced', 'balanced_subsample']),
    }


def lr_search_params(trial):
    """Logistic Regression search space, including class weighting."""
    return {
        'C': trial.suggest_float('C', 1e-4, 1e3, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
        'fit_intercept': trial.suggest_categorical(
            'fit_intercept', [True, False]),
        'class_weight': trial.suggest_categorical(
            'class_weight', [None, 'balanced']),
    }


def build_model(clf, params, n_jobs, seed):
    if clf == 'rf':
        estimator = RandomForestClassifier(
            **params, n_jobs=n_jobs, random_state=seed)
        return make_pipeline(SimpleImputer(strategy='median'), estimator)

    estimator = LogisticRegression(
        **params, solver='liblinear', max_iter=3000, random_state=seed)
    return make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        estimator,
    )


def tuned_tag(dataset, features, node_features, dim, clf, balanced, trials,
              seed, folds):
    """Make a run tag that records tuning for RF/LR without changing shared code."""
    tag = run_tag(dataset, features, node_features, dim, clf, balanced, 0,
                  seed, folds)
    marker = f'_seed{seed}_'
    return tag.replace(marker, f'_tune{trials}{marker}', 1)


def main():
    ap = argparse.ArgumentParser(
        description='Optuna tuning for Random Forest and Logistic Regression')
    ap.add_argument('-i', '--infile', required=True)
    ap.add_argument('-p', '--predfile', required=True)
    ap.add_argument('--features', default='concat',
                    choices=['gvfa', 'concat', 'bundle', 'rdkit'])
    ap.add_argument('--dim', type=int, default=500)
    ap.add_argument('--node-features', type=int, default=18, choices=[7, 18])
    ap.add_argument('--clf', required=True, choices=['rf', 'lr'])
    ap.add_argument('--trials', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--folds', type=int, default=10)
    ap.add_argument('--group', default='')
    ap.add_argument('--cache-dir', default='cache')
    ap.add_argument('--results-dir', default='results_other_classifier_tuning')
    ap.add_argument('--pval-dir', default=None)
    ap.add_argument(
        '--n-jobs', type=int,
        default=int(os.environ.get('OMP_NUM_THREADS', 1)),
        help='classifier and torch CPU threads')
    args = ap.parse_args()

    if args.trials < 1:
        raise SystemExit('--trials must be at least 1')

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    torch.set_num_threads(args.n_jobs)
    started_at = time.time()
    name = dataset_name_from_path(args.infile)

    train_ds, df_train = load_full(args.infile)
    test_ds, df_test = load_full(args.predfile)
    y_train, y_test = labels_of(df_train), labels_of(df_test)
    feasible = check_feasible(y_train, EPS_LEVELS, verbose=True)

    need_gvfa = args.features != 'rdkit'
    need_desc = args.features != 'gvfa'
    E_tr = E_te = D_tr = D_te = None
    if need_gvfa:
        E_tr = gvfa_embeddings(
            train_ds, name, args.dim, args.cache_dir, 'train',
            df_train['smiles'], args.node_features)
        E_te = gvfa_embeddings(
            test_ds, name, args.dim, args.cache_dir, 'test',
            df_test['smiles'], args.node_features)
    if need_desc:
        D_tr = rdkit_descriptors(df_train, name, args.cache_dir, 'train')
        D_te = rdkit_descriptors(df_test, name, args.cache_dir, 'test')

    X_tr, X_te, bundle_fn = assemble(
        args.features, D_tr, D_te, E_tr, E_te, args.dim, args.seed)
    if X_tr is not None:
        print(f'features {args.features}: X_train {X_tr.shape} '
              f'X_test {X_te.shape}')

    try:
        splits = get_splits(
            name, y_train, seed=args.seed, n_folds=args.folds)
    except ValueError as exc:
        print(f'SKIP {name}: {exc}')
        return

    n_per_class = {c: int((y_train == c).sum()) for c in (0, 1)}

    # Tune on fold 0's fit rows, with disjoint inner fit/calibration sets.
    f0, tune_eval_rows, _ = fold_indices(splits, 0)
    inner_fit, inner_cal = train_test_split(
        f0, test_size=0.25, stratify=y_train[f0],
        random_state=args.seed + 1)
    Xt = X_tr if X_tr is not None else bundle_fn(inner_fit)[0]

    def objective(trial):
        params = (rf_search_params(trial) if args.clf == 'rf'
                  else lr_search_params(trial))
        model = build_model(args.clf, params, args.n_jobs, args.seed)
        model.fit(Xt[inner_fit], y_train[inner_fit])

        p_cal = model.predict_proba(Xt[inner_cal])
        p_eval = model.predict_proba(Xt[tune_eval_rows])
        cal = {
            c: 1.0 - p_cal[y_train[inner_cal] == c, c]
            for c in (0, 1)
        }
        n_cal = {
            c: int((y_train[inner_cal] == c).sum())
            for c in (0, 1)
        }
        pvals = cross_conformal_pvalues([cal], [1.0 - p_eval], n_cal)
        metrics = cp_metrics(pvals, y_train[tune_eval_rows], 0.2)
        return metrics['eff'] - (0.0 if metrics['valid'] else 1.0)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials)
    best_params = dict(study.best_params)
    print(f'optuna best {study.best_value:.4f}: {best_params}')

    tune_info = {
        'n_trials': args.trials,
        'best_value': study.best_value,
        'best_params': best_params,
        'objective_confidence': 0.80,
    }

    # Refit the winning configuration in every cross-conformal fold.
    fold_cal, fold_test = [], []
    model = None
    for i in range(splits['n_folds']):
        fit_rows, tune_rows, cal_rows = fold_indices(splits, i)
        if bundle_fn is not None:
            X_tr, X_te = bundle_fn(fit_rows)

        model = build_model(args.clf, best_params, args.n_jobs, args.seed)
        train_rows = np.concatenate([fit_rows, tune_rows])
        model.fit(X_tr[train_rows], y_train[train_rows])

        p_cal = model.predict_proba(X_tr[cal_rows])
        fold_cal.append({
            c: 1.0 - p_cal[y_train[cal_rows] == c, c]
            for c in (0, 1)
        })
        fold_test.append(1.0 - model.predict_proba(X_te))
        print(f'  fold {i + 1}/{splits["n_folds"]} done')

    pvals = cross_conformal_pvalues(fold_cal, fold_test, n_per_class)

    group = args.group or 'nogroup'
    dataset = short_dataset_name(args.infile)
    balanced = best_params.get('class_weight') is not None
    tag = tuned_tag(
        dataset, args.features, args.node_features, args.dim, args.clf,
        balanced, args.trials, args.seed, args.folds)

    root = Path(args.results_dir)
    metrics_dir = root / 'metrics' / group
    pval_dir = (Path(args.pval_dir) if args.pval_dir
                else root / 'pvalues' / group)
    config_dir = root / 'configs' / group
    for directory in (metrics_dir, pval_dir, config_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_id = new_run_id(tag, started_at, metrics_dir)
    started = time.strftime(
        '%Y-%m-%d %H:%M:%S', time.localtime(started_at))
    runtime = round(time.time() - started_at, 1)
    clf_params = classifier_params(model)

    pval_path = pval_dir / f'{run_id}_p-values.csv'
    pd.DataFrame({
        'id': df_test['id'].astype(str),
        '0': pvals[:, 0],
        '1': pvals[:, 1],
        'class': y_test,
        'model': tag,
        'run_id': run_id,
    }).to_csv(pval_path, sep='\t', index=False)

    rows = []
    for eps in EPS_LEVELS:
        metrics = cp_metrics(pvals, y_test, eps)
        metrics.update(
            run_id=run_id, tag=tag, started=started,
            group=args.group, dataset=dataset, dataset_file=name,
            features=args.features,
            dim=args.dim if args.features != 'rdkit' else 0,
            node_features=(args.node_features
                           if args.features != 'rdkit' else 0),
            clf=args.clf, seed=args.seed, folds=args.folds,
            tuned=True, n_trials=args.trials, balance=balanced,
            eval_metric='', n_train=len(y_train),
            n_active_train=n_per_class[1], n_test=len(y_test),
            n_active_test=int(y_test.sum()),
            n_cal_active=n_per_class[1],
            min_cal_needed=min_calib_per_class(eps),
            feasible=feasible[eps], runtime_s=runtime,
            clf_params=json.dumps(
                clf_params, sort_keys=True, default=str))
        rows.append(metrics)
        print(
            f'  conf {metrics["conf"]:.2f}  eff {metrics["eff"]:.3f}  '
            f'err_act {metrics["err_active"]:.3f}  '
            f'err_inact {metrics["err_inactive"]:.3f}  '
            f'valid {metrics["valid"]}  feasible {metrics["feasible"]}')

    metrics_frame = pd.DataFrame(rows)
    metrics_path = metrics_dir / f'{run_id}.csv'
    metrics_frame.to_csv(metrics_path, index=False)

    config = {
        'run_id': run_id,
        'tag': tag,
        'started': started,
        'runtime_s': runtime,
        'dataset': dataset,
        'train_file': str(args.infile),
        'test_file': str(args.predfile),
        'args': vars(args),
        'classifier_params': clf_params,
        'tuning': tune_info,
        'eps_levels': list(EPS_LEVELS),
        'feasible': {str(k): bool(v) for k, v in feasible.items()},
        'n_train': len(y_train),
        'n_active_train': n_per_class[1],
        'n_test': len(y_test),
        'n_active_test': int(y_test.sum()),
        'outputs': {
            'metrics': str(metrics_path),
            'pvalues': str(pval_path),
        },
        'host': platform.node(),
        'pbs_jobid': os.environ.get('PBS_JOBID', ''),
        'command': ' '.join(sys.argv),
        'versions': package_versions(),
    }
    config_path = config_dir / f'{run_id}.json'
    config_path.write_text(json.dumps(config, indent=2, default=str))

    print(
        f'saved  metrics {metrics_path}\n'
        f'       p-values {pval_path}\n'
        f'       config  {config_path}')


if __name__ == '__main__':
    main()
