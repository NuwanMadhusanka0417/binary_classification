"""
result_naming.py -- result file names, shared by Run_experiment.py and
nci/make_commands.py so both always agree on what a run is called.

No heavy imports here: make_commands.py uses it to skip finished runs.
"""
import re
from pathlib import Path


def short_dataset_name(path):
    """Readable dataset id for result file names. Cache and split files keep
    the full file name as their key, so existing caches/splits stay valid."""
    base = Path(path).name
    for pat, fmt in ((r'^(AID_\d+)', '{0}'),                      # ED / msc_new
                     (r'^tox21\.data_([^.]+)\.txt', 'tox21_{0}'),  # tox21
                     (r'^HTSFP10\.data_(\d+)', 'HTSFP10_{0}')):    # kinase
        m = re.match(pat, base)
        if m:
            return fmt.format(m.group(1))
    return re.sub(r'[^A-Za-z0-9_-]+', '_', Path(path).stem)[:60]


def run_tag(dataset, features, node_features, dim, clf, balance, tune, seed,
            folds):
    """Settings part of a run's file name (no time stamp).

    <dataset>_<features>[_nf<N>_dim<D>]_<clf>[_bal][_tune<T>]_seed<S>_k<K>
    """
    parts = [dataset, features]
    if features != 'rdkit':                      # rdkit uses no encoder settings
        parts += [f'nf{node_features}', f'dim{dim}']
    parts.append(clf)
    if balance:
        parts.append('bal')
    if tune and clf == 'xgb':                    # --tune only acts on xgb
        parts.append(f'tune{tune}')
    parts += [f'seed{seed}', f'k{folds}']
    return '_'.join(parts)
