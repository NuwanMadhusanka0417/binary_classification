# import pandas as pd, torch
# from torch_geometric.data import InMemoryDataset, Data
# from rdkit import Chem
# from sklearn.model_selection import ShuffleSplit, KFold

# from rdkit import RDLogger
# RDLogger.DisableLog("rdApp.*")
# # Map RDKit bond types to a single integer (edge_attr is 1D like ZINC prints)
# _BOND2ID = {
#     Chem.BondType.SINGLE: 0,
#     Chem.BondType.DOUBLE: 1,
#     Chem.BondType.TRIPLE: 2,
#     Chem.BondType.AROMATIC: 3,
# }

# def smiles_to_data(smi, yval):
#     mol = Chem.MolFromSmiles(smi)
#     if mol is None or mol.GetNumAtoms() == 0:
#         return None

#     # x: [num_nodes, 1] (long) — single integer per atom (e.g., atomic number)
#     x = torch.tensor([[a.GetAtomicNum()] for a in mol.GetAtoms()], dtype=torch.long)

#     # Edges (both directions) and 1D edge_attr (long)
#     src, dst, eattr = [], [], []
#     for b in mol.GetBonds():
#         u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
#         t = _BOND2ID.get(b.GetBondType(), 0)
#         # add both directions
#         src += [u, v]
#         dst += [v, u]
#         eattr += [t, t]

#     edge_index = torch.tensor([src, dst], dtype=torch.long)
#     edge_attr  = torch.tensor(eattr, dtype=torch.long)  # shape [E], same as ZINC print

#     # y: [1] (float)
#     y = torch.tensor([float(yval)], dtype=torch.float)

#     return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# class ZINCLikeCSV(InMemoryDataset):
#     def __init__(self, csv_path, smiles_col="smiles_canon", target_col="LogS"):
#         df = pd.read_csv(csv_path)
#         super().__init__('.')
#         graphs = []
#         for smi, y in zip(df[smiles_col], df[target_col]):
#             g = smiles_to_data(smi, y)
#             if g is not None:
#                 graphs.append(g)
#         self.data, self.slices = self.collate(graphs)

# def load_data():
#   dataset_test  = ZINCLikeCSV("final_data/final_unique_test.csv")
#   dataset_train  = ZINCLikeCSV("final_data/final_unique_train_fixed.csv")

#   return dataset_train, dataset_test #, gl_train, 



# class ZINCLikeCSV2(InMemoryDataset):
#     def __init__(self, csv_path, smiles_col="smiles", target_col="target"):
#        #assumes UN data format (3 tab separated columns) id, target, smiles
#         df = pd.read_csv(csv_path, sep='\t', header = 0, index_col = None)

#         try:
#             labels = df['id'].values
#         except:
#             df.rename(columns={ df.columns[0]: "id" }, inplace = True)
#             labels = df['id'].values
#             print("/////////////////////// renaming column 1 to 'id' ///////////////////////")

#         try:
#             df.loc[df['target'] < 0, 'target'] = 0
#         except:
#             df.rename(columns={ df.columns[1]: "target" }, inplace = True)
#             df.loc[df['target'] < 0, 'target'] = 0
#             print("/////////////////////// renaming column 2 to 'target' ///////////////////////")

#         try:
#             labels = df['smiles'].values
#         except:
#             df.rename(columns={ df.columns[2]: "smiles" }, inplace = True)
#             print("/////////////////////// renaming column 2 to 'smiles' ///////////////////////")

#         super().__init__('.')
#         graphs = []
#         for smi, y in zip(df[smiles_col], df[target_col]):
#             g = smiles_to_data(smi, y)
#             if g is not None:
#                 graphs.append(g)
#         self.data, self.slices = self.collate(graphs)

# def load_data2(infile, predfile):
#   dataset_test  = ZINCLikeCSV2(predfile)
#   dataset_train  = ZINCLikeCSV2(infile)

#   return dataset_train, dataset_test #, gl_train, 


# class DataLikeCSV(InMemoryDataset):
#     def __init__(self, csv_path, task, smiles_col="smiles", target_col="target"):
#         #assumes UN data format (3 tab separated columns) id, target, smiles
#         df = pd.read_csv(csv_path, sep='\t', header = 0, index_col = None)

#         try:
#             labels = df['id'].values
#         except:
#             df.rename(columns={ df.columns[0]: "id" }, inplace = True)
#             labels = df['id'].values
#             print("/////////////////////// renaming column 1 to 'id' ///////////////////////")

#         try:
#             df.loc[df['target'] < 0, 'target'] = 0
#         except:
#             df.rename(columns={ df.columns[1]: "target" }, inplace = True)
#             df.loc[df['target'] < 0, 'target'] = 0
#             print("/////////////////////// renaming column 2 to 'target' ///////////////////////")

#         try:
#             labels = df['smiles'].values
#         except:
#             df.rename(columns={ df.columns[2]: "smiles" }, inplace = True)
#             print("/////////////////////// renaming column 3 to 'smiles' ///////////////////////")

#         if task == 'train':
#             super().__init__('.')

#             df = df.sample(frac=1, random_state=42)
#             nrows = df.shape[0]
#             nrowstr = int(0.8*nrows)
#             df_tr = df[:nrowstr]
# #            df_tr.to_csv('df_tr.txt', sep = '\t', header=True, index=False) #UN
#             df_val = df[nrowstr:]

#             graphs = []
#             for smi, y in zip(df_tr[smiles_col], df_tr[target_col]):
#                 g = smiles_to_data(smi, y)
#                 if g is not None:
#                     graphs.append(g)
#             self.data, self.slices = self.collate(graphs)

#         if task == 'val':
#             super().__init__('.')

#             df = df.sample(frac=1, random_state=42)
#             nrows = df.shape[0]
#             nrowstr = int(0.8*nrows)
#             df_tr = df[:nrowstr]
#             df_val = df[nrowstr:]
# #            df_val.to_csv('df_val.txt', sep = '\t', header=True, index=False) #UN

#             graphs = []
#             for smi, y in zip(df_val[smiles_col], df_val[target_col]):
#                 g = smiles_to_data(smi, y)
#                 if g is not None:
#                     graphs.append(g)
#             del df_tr, df_val
#             self.data, self.slices = self.collate(graphs)

#         if task == 'test':
#             super().__init__('.')
#             graphs = []
#             for smi, y in zip(df[smiles_col], df[target_col]):
#                 g = smiles_to_data(smi, y)
#                 if g is not None:
#                     graphs.append(g)
#             self.data, self.slices = self.collate(graphs)

# def load_undata(datafile,task):
#   dataset  = DataLikeCSV(datafile, task)
#   return dataset

# def df_read_csv(csv_path):
#         df = pd.read_csv(csv_path, sep='\t', header = 0, index_col = None)

#         try:
#             labels = df['id'].values
#         except:
#             df.rename(columns={ df.columns[0]: "id" }, inplace = True)
#             labels = df['id'].values
#             print("/////////////////////// renaming column 1 to 'id' ///////////////////////")

#         try:
#             df.loc[df['target'] < 0, 'target'] = 0
#         except:
#             df.rename(columns={ df.columns[1]: "target" }, inplace = True)
#             df.loc[df['target'] < 0, 'target'] = 0
#             print("/////////////////////// renaming column 2 to 'target' ///////////////////////")

#         try:
#             labels = df['smiles'].values
#         except:
#             df.rename(columns={ df.columns[2]: "smiles" }, inplace = True)
#             print("/////////////////////// renaming column 3 to 'smiles' ///////////////////////")

#         return df


'''
src/load_data.py

Loading and splitting for the GVFA conformal-prediction experiments.

DESIGN
------
A loader reads a file and returns graphs. It does not split. Splitting happens
once per (dataset, seed), is written to disk, and is reused by every one of the
54 feature/classifier configurations -- so differences between configurations
are attributable to the configuration and not to split noise.

Four roles, not three. `val` was doing two jobs, which is where the leak was:

    fit    -> fits the classifier                  may influence the model
    tune   -> early stopping, Optuna               may influence the model
    calib  -> nonconformity scores -> p-values     MUST NOT influence the model
    test   -> final reported numbers               MUST NOT influence the model

fit + tune together are the "proper training set".

CROSS-CONFORMAL
---------------
A Mondrian p-value is (count + 1) / (n_c + 1), so the smallest achievable
p-value for class c is 1/(n_c + 1). To exclude a label at significance eps you
need n_c >= 1/eps - 1:

    confidence 0.80 (eps 0.20) -> >= 4  calibration points of that class
    confidence 0.90 (eps 0.10) -> >= 9
    confidence 0.95 (eps 0.05) -> >= 19

AID_743241 has 14 actives in 6230 compounds. A single 80/20 calibration split
leaves ~2 actives, so a singleton prediction is arithmetically impossible.
make_splits() therefore uses k-fold cross-conformal: every training point acts
as calibration exactly once, so n_cal is the whole training set rather than 20%
of it. Use min_calib_per_class() to check a dataset before trusting its numbers.

smiles_to_data() is UNCHANGED -- the molecule -> graph conversion your existing
results were built on is untouched.
'''

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import StratifiedKFold, train_test_split

RDLogger.DisableLog("rdApp.*")

# Map RDKit bond types to a single integer (edge_attr is 1D like ZINC prints)
_BOND2ID = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}

REQUIRED_COLS = ('id', 'target', 'smiles')
_SPLIT_TASKS = {'train', 'val'}
_FULL_TASKS = {'test', 'all'}


# ===========================================================================
# molecule -> graph   (UNCHANGED)
# ===========================================================================
def smiles_to_data(smi, yval):
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    # x: [num_nodes, 1] (long) -- single integer per atom (atomic number)
    x = torch.tensor([[a.GetAtomicNum()] for a in mol.GetAtoms()], dtype=torch.long)

    # Edges (both directions) and 1D edge_attr (long)
    src, dst, eattr = [], [], []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        t = _BOND2ID.get(b.GetBondType(), 0)
        src += [u, v]
        dst += [v, u]
        eattr += [t, t]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(eattr, dtype=torch.long)
    y = torch.tensor([float(yval)], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ===========================================================================
# shared helpers (one implementation instead of four copies)
# ===========================================================================
def _normalise_columns(df, path):
    """UN format: 3 tab-separated columns id, target, smiles.

    Renames positionally only where the proper name is absent, and fails loudly
    instead of silently clobbering a column (the old bare `except:` would rename
    column 1 to 'target' if the target column happened to be a string dtype).
    """
    if df.shape[1] < 3:
        raise ValueError(f"{path}: expected >=3 columns (id, target, smiles), "
                         f"got {df.shape[1]}: {list(df.columns)}")

    original = list(df.columns)
    for pos, name in enumerate(REQUIRED_COLS):
        if name not in df.columns:
            print(f"{path}: renaming column {pos + 1} ({original[pos]!r}) -> {name!r}")
            df = df.rename(columns={original[pos]: name})

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: columns {missing} still missing after renaming")

    # negative label (-1) means inactive -> 0
    df['target'] = pd.to_numeric(df['target'], errors='raise')
    df.loc[df['target'] < 0, 'target'] = 0

    bad = set(df['target'].unique()) - {0, 1}
    if bad:
        raise ValueError(f"{path}: target must be binary after mapping, found {sorted(bad)}")

    return df


def _build_graphs(df, smiles_col, target_col, path):
    """Build graphs and return the rows that survived.

    Returns (graphs, df_kept) with len(graphs) == len(df_kept) and graphs[i]
    corresponding to df_kept.iloc[i]. This alignment is what stops predicted
    probabilities drifting against ids when a SMILES fails to parse.
    """
    graphs, keep = [], []
    for i, (smi, y) in enumerate(zip(df[smiles_col], df[target_col])):
        g = smiles_to_data(str(smi), y)
        if g is not None:
            graphs.append(g)
            keep.append(i)

    dropped = len(df) - len(keep)
    if dropped:
        print(f"{path}: dropped {dropped} unparseable SMILES")

    return graphs, df.iloc[keep].reset_index(drop=True)


def dataset_name_from_path(path):
    """Stable short name used to key split files and embedding caches."""
    return Path(path).name


# ===========================================================================
# the loader
# ===========================================================================
class DataLikeCSV(InMemoryDataset):
    """UN-format loader.

        task='test' / 'all'  -> the whole file, unsplit     <- use this
        task='train'         -> first 80% of a shuffle      (legacy)
        task='val'           -> last  20% of same shuffle   (legacy)

    After construction `self.df_used` holds exactly the rows that became
    graphs, in order, so df_used.iloc[i] matches dataset[i].

    The train/val branches reproduce the original unstratified seed-42 split so
    old runs still reproduce. New work should use task='all' and split with
    make_splits().
    """

    def __init__(self, csv_path, task, smiles_col="smiles", target_col="target",
                 seed=42, train_frac=0.8, stratify=False):
        if task not in _SPLIT_TASKS | _FULL_TASKS:
            raise ValueError(f"task must be one of {sorted(_SPLIT_TASKS | _FULL_TASKS)}, "
                             f"got {task!r}")

        df = pd.read_csv(csv_path, sep='\t', header=0, index_col=None)
        df = _normalise_columns(df, csv_path)

        super().__init__('.')

        if task in _SPLIT_TASKS:
            if stratify:
                tr_idx, val_idx = train_test_split(
                    np.arange(len(df)), train_size=train_frac,
                    stratify=df['target'].to_numpy(), random_state=seed)
                df = df.iloc[sorted(tr_idx if task == 'train' else val_idx)]
            else:
                df = df.sample(frac=1, random_state=seed)
                cut = int(train_frac * len(df))
                df = df.iloc[:cut] if task == 'train' else df.iloc[cut:]
            df = df.reset_index(drop=True)

        self.task = task
        self.source_path = str(csv_path)
        graphs, self.df_used = _build_graphs(df, smiles_col, target_col, csv_path)
        self.data, self.slices = self.collate(graphs)

    def __repr__(self):
        n_pos = int((self.df_used['target'] == 1).sum())
        return (f"DataLikeCSV(task={self.task!r}, n={len(self.df_used)}, "
                f"active={n_pos})")


def load_full(csv_path, verbose=True):
    """Load a whole file WITHOUT splitting it.

    Returns (dataset, df) with len(dataset) == len(df) and dataset[i]
    corresponding to df.iloc[i]. This is the entry point for the new protocol:
    load train and test once, build embeddings, then split with make_splits().
    """
    dataset = DataLikeCSV(csv_path, 'all')
    df = dataset.df_used
    if verbose:
        n_pos = int((df['target'] == 1).sum())
        print(f"{csv_path}: {len(df)} molecules, {n_pos} active "
              f"({n_pos / max(len(df), 1):.3f})")
    return dataset, df


def labels_of(df):
    """Integer label vector aligned to the graphs from load_full()."""
    return df['target'].to_numpy().astype(int)


# ===========================================================================
# splits -- generated once per (dataset, seed), persisted, shared by all configs
# ===========================================================================
def min_calib_per_class(eps):
    """Calibration points of a class needed to reach significance eps.
    p_min = 1/(n_c + 1) <= eps  =>  n_c >= 1/eps - 1."""
    return int(np.ceil(1.0 / eps - 1.0))


def check_feasible(y, eps_levels=(0.3, 0.2, 0.1, 0.05), n_folds=10, verbose=True):
    """Which confidence levels this dataset can support under cross-conformal.

    Under k-fold cross-conformal every training point is calibration exactly
    once, so n_cal per class is the full per-class count.
    """
    y = np.asarray(y).astype(int)
    n_act, n_inact = int((y == 1).sum()), int((y == 0).sum())
    out = {}
    for eps in eps_levels:
        need = min_calib_per_class(eps)
        out[eps] = (n_act >= need) and (n_inact >= need)
        if verbose and not out[eps]:
            print(f"  conf {1 - eps:.2f}: NOT FEASIBLE -- needs {need} per class, "
                  f"have {n_act} active / {n_inact} inactive")
    return out


def make_splits(y, n_folds=10, tune_frac=0.2, seed=42):
    """Cross-conformal folds over the training set.

    For each fold: `calib` is the held-out stratified fold; the remainder splits
    into `fit` (classifier) and `tune` (early stopping / Optuna). Every training
    point serves as calibration exactly once.

    Returns a dict that is JSON-serialisable.
    """
    y = np.asarray(y).astype(int)
    n_min = min(int((y == 0).sum()), int((y == 1).sum()))
    if n_folds > n_min:
        raise ValueError(f"n_folds={n_folds} exceeds the {n_min} members of the "
                         f"smaller class; StratifiedKFold cannot split this")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for rest, calib in skf.split(np.zeros(len(y)), y):
        fit, tune = train_test_split(
            rest, test_size=tune_frac, stratify=y[rest], random_state=seed)
        folds.append({'fit': fit.tolist(),
                      'tune': tune.tolist(),
                      'calib': calib.tolist()})

    return {'n': int(len(y)), 'n_active': int(y.sum()),
            'n_folds': int(n_folds), 'tune_frac': tune_frac, 'seed': int(seed),
            'folds': folds}


def split_path(dataset_name, seed, out_dir='splits'):
    return Path(out_dir) / f'{dataset_name}_seed{seed}.json'


def get_splits(dataset_name, y, seed=42, n_folds=10, tune_frac=0.2,
               out_dir='splits', verbose=True):
    """Load the saved splits if they exist, otherwise create and save them.

    Every configuration calls this and receives the identical split, so a
    difference between two configurations is attributable to the configuration.
    """
    p = split_path(dataset_name, seed, out_dir)
    if p.exists():
        s = json.loads(p.read_text())
        if s['n'] != len(y):
            raise ValueError(f"{p}: saved split has n={s['n']} but the data has "
                             f"{len(y)} rows. The dropped-SMILES count changed -- "
                             f"delete the split file and regenerate.")
        if verbose:
            print(f"splits: reusing {p} ({s['n_folds']} folds)")
        return s

    s = make_splits(y, n_folds=n_folds, tune_frac=tune_frac, seed=seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s))
    if verbose:
        print(f"splits: wrote {p} ({s['n_folds']} folds, "
              f"{s['n_active']} active of {s['n']})")
    return s


def fold_indices(splits, i):
    """(fit, tune, calib) index arrays for fold i."""
    f = splits['folds'][i]
    return (np.asarray(f['fit']), np.asarray(f['tune']), np.asarray(f['calib']))


def split_train_calib_es(y, calib_frac=0.2, es_frac=0.2, seed=42, verbose=True):
    """Single stratified three-way split: fit / tune / calib.

    Cheaper than cross-conformal and fine for quick iteration on the larger
    datasets, but on a dataset with few actives it leaves too few calibration
    actives to reach the target confidence -- check with check_feasible() first.
    Prefer make_splits() for anything reported.
    """
    y = np.asarray(y).astype(int)
    idx = np.arange(len(y))

    idx_prop, idx_cal = train_test_split(
        idx, test_size=calib_frac, stratify=y, random_state=seed)
    idx_fit, idx_tune = train_test_split(
        idx_prop, test_size=es_frac, stratify=y[idx_prop], random_state=seed)

    if verbose:
        for name, ix in (('fit', idx_fit), ('tune', idx_tune), ('calib', idx_cal)):
            print(f"  {name:>6}: {len(ix):6d}  active {int(y[ix].sum()):5d}")
        n_min = min(int((y[idx_cal] == 0).sum()), int((y[idx_cal] == 1).sum()))
        if n_min < 4:
            print(f"  WARNING: only {n_min} calibration points in the smaller class; "
                  f"confidence 0.8 needs at least {min_calib_per_class(0.2)}.")

    return idx_fit, idx_tune, idx_cal


# ===========================================================================
# legacy API -- kept so the existing scripts keep running unchanged
# ===========================================================================
class ZINCLikeCSV(InMemoryDataset):
    """Regression-era loader for final_data/*.csv. Nothing in the classification
    scripts imports this."""

    def __init__(self, csv_path, smiles_col="smiles_canon", target_col="LogS"):
        df = pd.read_csv(csv_path)
        super().__init__('.')
        graphs = []
        for smi, y in zip(df[smiles_col], df[target_col]):
            g = smiles_to_data(smi, y)
            if g is not None:
                graphs.append(g)
        self.data, self.slices = self.collate(graphs)


def load_data():
    dataset_test = ZINCLikeCSV("final_data/final_unique_test.csv")
    dataset_train = ZINCLikeCSV("final_data/final_unique_train_fixed.csv")
    return dataset_train, dataset_test


class ZINCLikeCSV2(InMemoryDataset):
    """Used by load_data2(). Same as DataLikeCSV(task='all')."""

    def __init__(self, csv_path, smiles_col="smiles", target_col="target"):
        df = pd.read_csv(csv_path, sep='\t', header=0, index_col=None)
        df = _normalise_columns(df, csv_path)
        super().__init__('.')
        self.source_path = str(csv_path)
        graphs, self.df_used = _build_graphs(df, smiles_col, target_col, csv_path)
        self.data, self.slices = self.collate(graphs)


def load_data2(infile, predfile):
    """Used by GVFA_with_Traditional_feature_class.py and ..._bundling.py."""
    dataset_test = ZINCLikeCSV2(predfile)
    dataset_train = ZINCLikeCSV2(infile)
    return dataset_train, dataset_test


def load_undata(datafile, task):
    """Used by GVFA_class_main.py. Reproduces the original split behaviour."""
    return DataLikeCSV(datafile, task)


def df_read_csv(csv_path):
    """Read a UN-format file and normalise its columns. No SMILES parsing, so
    this does NOT drop unparseable rows -- use load_full() when the dataframe
    must line up with an embedding matrix."""
    df = pd.read_csv(csv_path, sep='\t', header=0, index_col=None)
    return _normalise_columns(df, csv_path)