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
'''

import argparse
import json
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
from models.graphcnnVSA_Binding_FULL import GraphCNN

EPS_LEVELS = (0.30, 0.20, 0.10, 0.05)


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
    hvs = VSA_conversion(graphs, dim)
    model = GraphCNN(hvs[0].node_features.shape[1], 5, 1, 'sum', 'sum',
                     torch.device('cpu'), 10)
    with torch.no_grad():                       # encoder is never trained
        emb, _ = getEmbedding(model, torch.device('cpu'), hvs)
    X = emb.squeeze(0).numpy().astype(np.float32)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, X)
    return X


def rdkit_descriptors(df, name, cache_dir, split_tag):
    """The 96 RDKit descriptors, cached."""
    cache = Path(cache_dir) / f'{name}_{split_tag}_desc96.npy'
    if cache.exists():
        return np.load(cache)
    X = utilities.generate_desc_96(df['smiles']).to_numpy(dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, X)
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
def xgb_space(trial, pos_weight, eval_metric):
    return dict(
        n_estimators=2000, eval_metric=eval_metric, random_state=42, n_jobs=10,
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
                n_jobs=10, early_stopping_rounds=100, tree_method='hist',
                scale_pos_weight=pos_weight, learning_rate=0.03, max_depth=7,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0)


def build(clf, params, balance):
    if clf == 'xgb':
        return XGBClassifier(**params)
    if clf == 'rf':
        return RandomForestClassifier(
            n_estimators=500, n_jobs=10, random_state=42,     # was unseeded
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
    ap.add_argument('--out', default='results.csv')
    ap.add_argument('--pval-dir', default='pvalues')
    args = ap.parse_args()

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

    n_per_class = {c: int((y_train == c).sum()) for c in (0, 1)}
    pos_weight = n_per_class[0] / max(n_per_class[1], 1) if args.balance else 1.0

    # ---- optional tuning, on fold 0's fit rows only -----------------------
    params = xgb_default(pos_weight, args.eval_metric)
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
        params.update(study.best_params)

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
    nf_tag = '' if args.node_features == 7 or args.features == 'rdkit' \
        else f'_nf{args.node_features}'
    tag = (f'{name}_{args.features}{nf_tag}_dim{args.dim}_{args.clf}'
           f'_seed{args.seed}_k{args.folds}')

    Path(args.pval_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'id': df_test['id'].astype(str),
                  '0': pvals[:, 0], '1': pvals[:, 1],
                  'class': y_test, 'model': tag}
                 ).to_csv(Path(args.pval_dir) / f'{tag}_p-values.csv',
                          sep='\t', index=False)

    rows = []
    for eps in EPS_LEVELS:
        m = cp_metrics(pvals, y_test, eps)
        m.update(group=args.group, dataset=name, features=args.features,
                 dim=args.dim if args.features != 'rdkit' else 0,
                 node_features=args.node_features if args.features != 'rdkit' else 0,
                 clf=args.clf, seed=args.seed, folds=args.folds,
                 tuned=bool(args.tune), balance=args.balance,
                 n_train=len(y_train), n_active_train=n_per_class[1],
                 n_test=len(y_test), n_active_test=int(y_test.sum()),
                 n_cal_active=n_per_class[1],
                 min_cal_needed=min_calib_per_class(eps),
                 feasible=feasible[eps],
                 runtime_s=round(time.time() - t0, 1))
        rows.append(m)
        print(f'  conf {m["conf"]:.2f}  eff {m["eff"]:.3f}  '
              f'err_act {m["err_active"]:.3f}  err_inact {m["err_inactive"]:.3f}  '
              f'valid {m["valid"]}  feasible {m["feasible"]}')

    out = Path(args.out)
    pd.DataFrame(rows).to_csv(out, mode='a', index=False, header=not out.exists())
    print(f'appended {len(rows)} rows to {out}')


if __name__ == '__main__':
    main()