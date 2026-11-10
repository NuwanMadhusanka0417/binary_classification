#!/usr/bin/env python
'''
run_experiment.py -- one run of the GVFA conformal experiment.

Replaces GVFA_class_main.py, GVFA_with_Traditional_feature_class.py and
..._bundling.py, which were ~80% identical and had already drifted apart (the
shallow-copy bug exists in the first but not the other two, so GVFA-alone and
concatenation numbers came from different encoders).

    --features gvfa    GVFA hypervectors alone
    --features concat  96 RDKit descriptors || GVFA           <- best in the draft
    --features bundle  RP(96 descriptors) + GVFA, L2-normalised
    --features rdkit   96 RDKit descriptors alone  <- the Fig 1 baseline, which
                                                      no existing script produces

Protocol (this is what changes the numbers):
  * k-fold cross-conformal. Every training point is calibration exactly once,
    so n_cal is the whole training set instead of 20% of it. Needed because a
    single 80/20 split leaves ~2 actives on the smallest ED datasets, where a
    singleton prediction is then arithmetically impossible.
  * fit / tune / calib are three disjoint sets. Early stopping sees `tune`,
    never `calib`. That leak is the reason the draft's validity is optimistic.
  * splits are generated once per (dataset, seed) and shared by all 54
    configurations, so a difference between configurations is attributable to
    the configuration and not to split noise.
  * embeddings and descriptors are cached to .npy, which is what makes the
    Optuna grid affordable.

Usage:
  python run_experiment.py -i train.txt -p test.txt --features concat \
      --dim 500 --clf xgb --seed 42 --folds 10 [--tune 40] [--group tox21]
      [--node-features 18]

  --node-features 7   original 7 node features (src/create_graphs.py), default
  --node-features 18  the 18 Table 1 node features (src/create_graph_new.py),
                      computed on the original SMILES molecule. Only the input
                      to the encoder changes; projection, GraphCNN and the
                      conformal protocol are identical.

Outputs (one set of files per run, nothing is appended, so parallel cluster
jobs cannot corrupt each other):

  results/metrics/<group>/<run_id>.csv            4 rows, one per confidence level
  results/pvalues/<group>/<run_id>_p-values.csv   per-molecule p-values (test set)
  results/configs/<group>/<run_id>.json           every setting, classifier params,
                                                  package versions, host, PBS job

  run_id = <dataset>_<features>[_nf<N>_dim<D>]_<clf>[_bal][_tune<T>]_seed<S>_k<K>
           __<YYYYmmdd-HHMMSS>[_job<PBS id>]
  e.g.     AID_743122_gvfa_nf18_dim500_xgb_seed42_k10__20261109-185214_job123456

  python collect_results.py   merges results/metrics/**.csv into one table.
  --results-dir changes the root; --out additionally appends to one CSV (legacy).
'''

import argparse
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.random_projection import GaussianRandomProjection
from xgboost import XGBClassifier

from src.load_data import (load_full, labels_of, get_splits, fold_indices,
                           check_feasible, dataset_name_from_path,
                           min_calib_per_class)
from src.create_graphs import create_graph_list
from src.create_graph_new import create_graph_list as create_graph_list_18
from src.VSA_conversion import VSA_conversion
from src.embeddings import getEmbedding
from src import utilities
from src.result_naming import short_dataset_name, run_tag
from models.graphcnnVSA_Binding_FULL import GraphCNN

EPS_LEVELS = (0.30, 0.20, 0.10, 0.05)


# ===========================================================================
# result file naming
# ===========================================================================
EMBED_CHUNK = 2000   # molecules projected/encoded at once (multiple of getEmbedding's 100)


def save_npy_atomic(path, X):
    """Write to a temp file, then rename. Parallel tasks never see a half-written
    cache file, and a killed job never leaves a corrupt one behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    with open(tmp, 'wb') as f:
        np.save(f, X)
    os.replace(tmp, path)


def new_run_id(tag, started, metrics_dir):
    """tag + start time (+ PBS job id); never overwrites an earlier run."""
    stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime(started))
    job = os.environ.get('PBS_JOBID', '').split('.')[0]
    run_id = f'{tag}__{stamp}' + (f'_job{job}' if job else '')
    candidate, n = run_id, 1
    while (metrics_dir / f'{candidate}.csv').exists():
        candidate, n = f'{run_id}-{n}', n + 1
    return candidate


def classifier_params(model):
    """Parameters of the fitted classifier (last fold; all folds share them)."""
    est = model.steps[-1][1] if hasattr(model, 'steps') else model
    return {k: v for k, v in est.get_params(deep=False).items()
            if not callable(v)}


def package_versions():
    import sklearn, xgboost, rdkit
    out = {'python': platform.python_version(), 'numpy': np.__version__,
           'pandas': pd.__version__, 'torch': torch.__version__,
           'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__,
           'rdkit': rdkit.__version__}
    try:
        import torch_geometric
        out['torch_geometric'] = torch_geometric.__version__
    except ImportError:
        pass
    return out


# ===========================================================================
# cached feature blocks
# ===========================================================================
def gvfa_embeddings(dataset, name, dim, cache_dir, split_tag,
                    smiles=None, node_features=7):
    """GVFA hypervectors, cached. Graphs are rebuilt per call: VSA_conversion
    mutates node_features in place, so reusing a graph list across dimensions
    projects an already-projected matrix (the shallow-copy bug).

    node_features=7  : src/create_graphs.py (original, unchanged)
    node_features=18 : src/create_graph_new.py (Table 1), needs aligned `smiles`
    Only the node-feature input changes; projection and GraphCNN are the same."""
    nf_tag = '' if node_features == 7 else f'_nf{node_features}'   # 7 keeps old cache names
    cache = Path(cache_dir) / f'{name}_{split_tag}_gvfa{nf_tag}_dim{dim}.npy'
    if cache.exists():
        return np.load(cache)

    if node_features == 18:
        graphs = create_graph_list_18(dataset, smiles)
    else:
        graphs = create_graph_list(dataset)

    # Project and encode EMBED_CHUNK molecules at a time. Projecting the whole
    # set at once holds every atom as a D-wide vector (HTSFP10, D=10000: ~44 GB).
    # Same result: W is regenerated from the same fixed seed for every call of
    # VSA_conversion, and each molecule's embedding depends only on its own
    # graph (block-diagonal adjacency and per-graph pooling).
    # Chunks must be whole multiples of getEmbedding's batch of 100: then every
    # batch holds exactly the molecules it held before and the output is
    # byte-identical. Other sizes shift batch boundaries (~1e-8 rounding).
    if EMBED_CHUNK % 100:
        raise ValueError(f'EMBED_CHUNK={EMBED_CHUNK} must be a multiple of 100')
    model, parts = None, []
    for start in range(0, len(graphs), EMBED_CHUNK):
        hvs = VSA_conversion(graphs[start:start + EMBED_CHUNK], dim)
        if model is None:
            model = GraphCNN(hvs[0].node_features.shape[1], 5, 1, 'sum', 'sum',
                             torch.device('cpu'), 10)
        with torch.no_grad():                   # encoder is never trained
            emb, _ = getEmbedding(model, torch.device('cpu'), hvs)
        parts.append(emb.squeeze(0).numpy().astype(np.float32))
        for g in hvs:                           # free the D-wide node vectors
            g.node_features = None
    X = np.concatenate(parts, axis=0)

    save_npy_atomic(cache, X)
    return X


def rdkit_descriptors(df, name, cache_dir, split_tag):
    """The 96 RDKit descriptors, cached."""
    cache = Path(cache_dir) / f'{name}_{split_tag}_desc96.npy'
    if cache.exists():
        return np.load(cache)
    X = utilities.generate_desc_96(df['smiles']).to_numpy(dtype=np.float32)
    save_npy_atomic(cache, X)
    return X


def assemble(features, D_tr, D_te, E_tr, E_te, dim, seed):
    """Build the model input. `bundle` returns a callable fitted per fold,
    because its scaler and random projection must not see the calibration rows."""
    if features == 'gvfa':
        return E_tr, E_te, None
    if features == 'rdkit':
        return D_tr, D_te, None
    if features == 'concat':
        return (np.hstack([D_tr, E_tr]).astype(np.float32),
                np.hstack([D_te, E_te]).astype(np.float32), None)
    if features == 'bundle':
        def make(fit_rows):
            Dc = np.nan_to_num(D_tr, nan=np.nan, posinf=np.nan, neginf=np.nan)
            pipe = make_pipeline(
                SimpleImputer(strategy='median'),
                RobustScaler(),
                GaussianRandomProjection(n_components=dim, random_state=seed))
            pipe.fit(Dc[fit_rows])              # fitted on fit rows only
            A_tr = pipe.transform(Dc).astype(np.float32)
            A_te = pipe.transform(np.nan_to_num(D_te, nan=np.nan,
                                                posinf=np.nan,
                                                neginf=np.nan)).astype(np.float32)
            S_tr, S_te = A_tr + E_tr, A_te + E_te
            S_tr /= np.linalg.norm(S_tr, axis=1, keepdims=True) + 1e-8
            S_te /= np.linalg.norm(S_te, axis=1, keepdims=True) + 1e-8
            return S_tr, S_te
        return None, None, make
    raise ValueError(features)


# ===========================================================================
# classifiers
# ===========================================================================
N_JOBS = 10      # threads per classifier; set from --n-jobs in main()


def xgb_space(trial, pos_weight, eval_metric):
    return dict(
        n_estimators=2000, eval_metric=eval_metric, random_state=42, n_jobs=N_JOBS,
        early_stopping_rounds=100, tree_method='hist', scale_pos_weight=pos_weight,
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        min_child_weight=trial.suggest_int('min_child_weight', 1, 20),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bynode=trial.suggest_float('colsample_bynode', 0.2, 1.0),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-2, 100.0, log=True),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        gamma=trial.suggest_float('gamma', 0.0, 5.0))


def xgb_default(pos_weight, eval_metric):
    return dict(n_estimators=2000, eval_metric=eval_metric, random_state=42,
                n_jobs=N_JOBS, early_stopping_rounds=100, tree_method='hist',
                scale_pos_weight=pos_weight, learning_rate=0.03, max_depth=7,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0)


def build(clf, params, balance):
    if clf == 'xgb':
        return XGBClassifier(**params)
    if clf == 'rf':
        return RandomForestClassifier(
            n_estimators=500, n_jobs=N_JOBS, random_state=42,  # was unseeded
            class_weight='balanced_subsample' if balance else None)
    if clf == 'lr':
        return make_pipeline(                                  # was unscaled
            SimpleImputer(strategy='median'), StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=0,
                               class_weight='balanced' if balance else None))
    raise ValueError(clf)


def fit(model, clf, X, y, fit_rows, tune_rows):
    if clf == 'xgb':
        model.fit(X[fit_rows], y[fit_rows],
                  eval_set=[(X[tune_rows], y[tune_rows])], verbose=False)
    else:
        rows = np.concatenate([fit_rows, tune_rows])
        model.fit(X[rows], y[rows])
    return model


# ===========================================================================
# cross-conformal prediction (Vovk), class-conditional / Mondrian
# ===========================================================================
def cross_conformal_pvalues(fold_cal_scores, fold_test_scores, n_per_class):
    """Pooled-count cross-conformal p-values.

    For each fold m, count the calibration scores of class c that are at least
    as nonconforming as the test score produced BY THAT FOLD'S MODEL. Sum the
    counts across folds, then apply the +1 correction once against the total
    calibration count. Nonconformity = 1 - p_hat(c | x).
    """
    n_test = fold_test_scores[0].shape[0]
    counts = np.zeros((n_test, 2))
    for cal_c, test_s in zip(fold_cal_scores, fold_test_scores):
        for c in (0, 1):
            a_cal = cal_c[c]
            if len(a_cal):
                counts[:, c] += np.sum(a_cal[None, :] >= test_s[:, c][:, None], axis=1)
    pvals = np.zeros((n_test, 2))
    for c in (0, 1):
        pvals[:, c] = (counts[:, c] + 1.0) / (n_per_class[c] + 1.0)
    return pvals


def cp_metrics(pvals, y, eps):
    """Validity = per-class error rate <= eps. Efficiency = singleton rate."""
    pred = pvals > eps
    singleton = pred.sum(axis=1) == 1
    covered = pred[np.arange(len(y)), y]
    out = {'eps': eps, 'conf': round(1 - eps, 2), 'eff': float(singleton.mean())}
    for c, nm in ((0, 'inactive'), (1, 'active')):
        m = y == c
        out[f'err_{nm}'] = float(1 - covered[m].mean()) if m.sum() else np.nan
        out[f'eff_{nm}'] = float(singleton[m].mean()) if m.sum() else np.nan
    out['valid'] = bool(out['err_active'] <= eps and out['err_inactive'] <= eps)
    return out


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--infile', required=True)
    ap.add_argument('-p', '--predfile', required=True)
    ap.add_argument('--features', required=True,
                    choices=['gvfa', 'concat', 'bundle', 'rdkit'])
    ap.add_argument('--dim', type=int, default=500)
    ap.add_argument('--node-features', type=int, default=7, choices=[7, 18],
                    help='7 = original create_graphs.py; 18 = Table 1 features '
                         '(create_graph_new.py)')
    ap.add_argument('--clf', default='xgb', choices=['xgb', 'rf', 'lr'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--folds', type=int, default=10)
    ap.add_argument('--tune', type=int, default=0, help='Optuna trials (0 = off)')
    ap.add_argument('--balance', action='store_true')
    ap.add_argument('--eval-metric', default='aucpr', choices=['aucpr', 'logloss'])
    ap.add_argument('--group', default='')
    ap.add_argument('--cache-dir', default='cache')
    ap.add_argument('--results-dir', default='results',
                    help='root folder for metrics/, pvalues/ and configs/')
    ap.add_argument('--out', default=None,
                    help='optional: also append the metric rows to this one CSV')
    ap.add_argument('--pval-dir', default=None,
                    help='optional: override <results-dir>/pvalues/<group>')
    ap.add_argument('--n-jobs', type=int,
                    default=int(os.environ.get('OMP_NUM_THREADS', 10)),
                    help='threads for the classifier and torch '
                         '(default: $OMP_NUM_THREADS, else 10)')
    ap.add_argument('--prepare-only', action='store_true',
                    help='build and cache the feature blocks and the splits, '
                         'then stop (no classifier, no results). Used by the '
                         'cluster sweep so parallel runs only read caches.')
    args = ap.parse_args()

    global N_JOBS
    N_JOBS = args.n_jobs
    torch.set_num_threads(args.n_jobs)

    t0 = time.time()
    name = dataset_name_from_path(args.infile)

    # ---- data -------------------------------------------------------------
    train_ds, df_train = load_full(args.infile)
    test_ds, df_test = load_full(args.predfile)
    y_train, y_test = labels_of(df_train), labels_of(df_test)

    feasible = check_feasible(y_train, EPS_LEVELS, verbose=True)

    # ---- feature blocks (cached) -----------------------------------------
    need_gvfa = args.features != 'rdkit'
    need_desc = args.features != 'gvfa'
    E_tr = E_te = D_tr = D_te = None
    if need_gvfa:
        E_tr = gvfa_embeddings(train_ds, name, args.dim, args.cache_dir, 'train',
                               df_train['smiles'], args.node_features)
        E_te = gvfa_embeddings(test_ds, name, args.dim, args.cache_dir, 'test',
                               df_test['smiles'], args.node_features)
    if need_desc:
        D_tr = rdkit_descriptors(df_train, name, args.cache_dir, 'train')
        D_te = rdkit_descriptors(df_test, name, args.cache_dir, 'test')

    X_tr, X_te, bundle_fn = assemble(args.features, D_tr, D_te, E_tr, E_te,
                                     args.dim, args.seed)
    if X_tr is not None:
        print(f'features {args.features}: X_train {X_tr.shape}  X_test {X_te.shape}')

    # ---- splits (shared by every configuration) ---------------------------
    try:
        splits = get_splits(name, y_train, seed=args.seed, n_folds=args.folds)
    except ValueError as e:
        print(f'SKIP {name}: {e}')
        sys.exit(0)

    if args.prepare_only:
        print(f'prepared {name}: features={args.features} '
              f'nf={args.node_features} dim={args.dim}; splits ready')
        return

    n_per_class = {c: int((y_train == c).sum()) for c in (0, 1)}
    pos_weight = n_per_class[0] / max(n_per_class[1], 1) if args.balance else 1.0

    # ---- optional tuning, on fold 0's fit rows only -----------------------
    params = xgb_default(pos_weight, args.eval_metric)
    tune_info = None
    if args.tune and args.clf == 'xgb':
        import optuna
        from sklearn.model_selection import train_test_split
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        f0, t0_rows, _ = fold_indices(splits, 0)
        Xt = X_tr if X_tr is not None else bundle_fn(f0)[0]
        inner_fit, inner_cal = train_test_split(
            f0, test_size=0.25, stratify=y_train[f0], random_state=args.seed + 1)
        inner_fit, inner_es = train_test_split(
            inner_fit, test_size=0.2, stratify=y_train[inner_fit],
            random_state=args.seed + 1)

        def objective(trial):
            m = XGBClassifier(**xgb_space(trial, pos_weight, args.eval_metric))
            m.fit(Xt[inner_fit], y_train[inner_fit],
                  eval_set=[(Xt[inner_es], y_train[inner_es])], verbose=False)
            pc, pe = m.predict_proba(Xt[inner_cal]), m.predict_proba(Xt[t0_rows])
            cal = {c: 1 - pc[y_train[inner_cal] == c, c] for c in (0, 1)}
            npc = {c: int((y_train[inner_cal] == c).sum()) for c in (0, 1)}
            pv = cross_conformal_pvalues([cal], [1 - pe], npc)
            r = cp_metrics(pv, y_train[t0_rows], 0.2)
            return r['eff'] - (0.0 if r['valid'] else 1.0)   # invalid is disqualifying

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=args.seed))
        study.optimize(objective, n_trials=args.tune)
        print(f'optuna best {study.best_value:.4f}: {study.best_params}')
        # The trials searched colsample_bynode with colsample_bytree at its
        # XGBoost default of 1.0; the default dict sets 0.8. Reset it so the
        # model that runs is the one the winning trial actually evaluated.
        params['colsample_bytree'] = 1.0
        params.update(study.best_params)
        tune_info = {'n_trials': args.tune, 'best_value': study.best_value,
                     'best_params': study.best_params}

    # ---- cross-conformal loop --------------------------------------------
    fold_cal, fold_test = [], []
    for i in range(splits['n_folds']):
        f_fit, f_tune, f_cal = fold_indices(splits, i)
        if bundle_fn is not None:
            X_tr, X_te = bundle_fn(f_fit)        # refit per fold, no calib leak

        model = build(args.clf, params, args.balance)
        model = fit(model, args.clf, X_tr, y_train, f_fit, f_tune)

        p_cal = model.predict_proba(X_tr[f_cal])
        fold_cal.append({c: 1.0 - p_cal[y_train[f_cal] == c, c] for c in (0, 1)})
        fold_test.append(1.0 - model.predict_proba(X_te))
        print(f'  fold {i + 1}/{splits["n_folds"]} done')

    pvals = cross_conformal_pvalues(fold_cal, fold_test, n_per_class)

    # ---- report -----------------------------------------------------------
    group = args.group or 'nogroup'
    dataset = short_dataset_name(args.infile)
    tag = run_tag(dataset, args.features, args.node_features, args.dim, args.clf,
                  args.balance, args.tune, args.seed, args.folds)
    root = Path(args.results_dir)
    metrics_dir = root / 'metrics' / group
    pval_dir = Path(args.pval_dir) if args.pval_dir else root / 'pvalues' / group
    config_dir = root / 'configs' / group
    for d in (metrics_dir, pval_dir, config_dir):
        d.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id(tag, t0, metrics_dir)
    started = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))
    runtime = round(time.time() - t0, 1)
    tuned = tune_info is not None
    clf_params = classifier_params(model)

    pval_path = pval_dir / f'{run_id}_p-values.csv'
    pd.DataFrame({'id': df_test['id'].astype(str),
                  '0': pvals[:, 0], '1': pvals[:, 1],
                  'class': y_test, 'model': tag, 'run_id': run_id}
                 ).to_csv(pval_path, sep='\t', index=False)

    rows = []
    for eps in EPS_LEVELS:
        m = cp_metrics(pvals, y_test, eps)
        m.update(run_id=run_id, tag=tag, started=started,
                 group=args.group, dataset=dataset, dataset_file=name,
                 features=args.features,
                 dim=args.dim if args.features != 'rdkit' else 0,
                 node_features=args.node_features if args.features != 'rdkit' else 0,
                 clf=args.clf, seed=args.seed, folds=args.folds,
                 tuned=tuned, n_trials=args.tune if tuned else 0,
                 balance=args.balance,
                 eval_metric=args.eval_metric if args.clf == 'xgb' else '',
                 n_train=len(y_train), n_active_train=n_per_class[1],
                 n_test=len(y_test), n_active_test=int(y_test.sum()),
                 n_cal_active=n_per_class[1],
                 min_cal_needed=min_calib_per_class(eps),
                 feasible=feasible[eps],
                 runtime_s=runtime,
                 clf_params=json.dumps(clf_params, sort_keys=True, default=str))
        rows.append(m)
        print(f'  conf {m["conf"]:.2f}  eff {m["eff"]:.3f}  '
              f'err_act {m["err_active"]:.3f}  err_inact {m["err_inactive"]:.3f}  '
              f'valid {m["valid"]}  feasible {m["feasible"]}')

    metrics = pd.DataFrame(rows)
    metrics_path = metrics_dir / f'{run_id}.csv'
    metrics.to_csv(metrics_path, index=False)

    config = {'run_id': run_id, 'tag': tag, 'started': started,
              'runtime_s': runtime, 'dataset': dataset,
              'train_file': str(args.infile), 'test_file': str(args.predfile),
              'args': vars(args), 'classifier_params': clf_params,
              'tuning': tune_info, 'eps_levels': list(EPS_LEVELS),
              'feasible': {str(k): bool(v) for k, v in feasible.items()},
              'n_train': len(y_train), 'n_active_train': n_per_class[1],
              'n_test': len(y_test), 'n_active_test': int(y_test.sum()),
              'outputs': {'metrics': str(metrics_path), 'pvalues': str(pval_path)},
              'host': platform.node(), 'pbs_jobid': os.environ.get('PBS_JOBID', ''),
              'command': ' '.join(sys.argv), 'versions': package_versions()}
    config_path = config_dir / f'{run_id}.json'
    config_path.write_text(json.dumps(config, indent=2, default=str))

    print(f'saved  metrics {metrics_path}\n       p-values {pval_path}\n'
          f'       config  {config_path}')

    if args.out:                                   # optional legacy single CSV
        out = Path(args.out)
        if out.exists():
            header = out.open().readline().strip().split(',')
            if header != list(metrics.columns):
                print(f'NOT appended to {out}: its columns differ from this run '
                      f'(written by an older version). Rename it and rerun, or '
                      f'use collect_results.py instead.')
                return
        metrics.to_csv(out, mode='a', index=False, header=not out.exists())
        print(f'appended {len(rows)} rows to {out}')


if __name__ == '__main__':
    main()