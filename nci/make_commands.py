#!/usr/bin/env python
'''
make_commands.py -- write the command lists for a full sweep on NCI Gadi.

Three stages, each a plain text file with one shell command per line, run by
nci-parallel inside nci/sweep.pbs:

  stage1_prepare.txt  one line per dataset: RDKit descriptors + the shared
                      cross-conformal split. Must finish before stage 2/3 so
                      parallel tasks never race to create the same split file.
  stage2_embed.txt    one line per dataset x node-features x dim: build and
                      cache the GVFA embeddings once.
  stage3_runs.txt     one line per experiment (dataset x features x nf x dim x
                      classifier x balance). Reads the caches only.

Runs whose metrics file already exists in results/metrics/<group>/<tag>__*.csv
are left out, so re-submitting the same job continues where it stopped.
Lines are ordered largest dataset / dimension first, so the slowest tasks start
first and the end of the job is not one long straggler.

Usage (from the project root):
  python nci/make_commands.py --groups ED tox21 HTSFP10 --tune 50
  python nci/make_commands.py --groups ED --tune 0 --dims 500 1000   # small test
'''
import argparse
import itertools
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.result_naming import short_dataset_name, run_tag   # noqa: E402

GROUP_FOLDERS = {'ED': 'msc_new', 'tox21': 'tox21.data', 'HTSFP10': 'HTSFP10'}
TRAIN_SUFFIX, TEST_SUFFIX = '.tr_0.2_1.txt', '.te_0.2_1.txt'


def dataset_pairs(group, folder):
    """(train, test) file pairs in one group folder, matched by name."""
    pairs = []
    for tr in sorted((ROOT / folder).glob(f'*{TRAIN_SUFFIX}')):
        te = tr.with_name(tr.name[:-len(TRAIN_SUFFIX)] + TEST_SUFFIX)
        if te.exists():
            pairs.append((tr, te))
        else:
            print(f'WARNING {group}: no test file for {tr.name}', file=sys.stderr)
    return pairs


def finished(results_dir, group, tag):
    return any((results_dir / 'metrics' / group).glob(f'{tag}__*.csv'))


# cache / split names must match Run_experiment.py and src/load_data.split_path
def prepared(cache_dir, name, seed):
    return (all((cache_dir / f'{name}_{s}_desc96.npy').exists() for s in ('train', 'test'))
            and (ROOT / 'splits' / f'{name}_seed{seed}.json').exists())


def embedded(cache_dir, name, nf, dim):
    nf_tag = '' if nf == 7 else f'_nf{nf}'
    return all((cache_dir / f'{name}_{s}_gvfa{nf_tag}_dim{dim}.npy').exists()
               for s in ('train', 'test'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', nargs='+', default=['ED', 'tox21', 'HTSFP10'],
                    help='any of: ' + ' '.join(GROUP_FOLDERS))
    ap.add_argument('--features', nargs='+', default=['gvfa', 'concat', 'bundle'],
                    help='GVFA-based feature sets (rdkit baseline: --rdkit)')
    ap.add_argument('--rdkit', type=int, default=1,
                    help='1 = also run the rdkit-only baseline per classifier')
    ap.add_argument('--dims', nargs='+', type=int,
                    default=[100, 500, 1000, 2000, 5000, 10000])
    ap.add_argument('--node-features', nargs='+', type=int, default=[7, 18])
    ap.add_argument('--clfs', nargs='+', default=['xgb', 'rf', 'lr'])
    ap.add_argument('--balance', nargs='+', type=int, default=[0],
                    help='0 = off, 1 = on; "0 1" runs both')
    ap.add_argument('--tune', type=int, default=0,
                    help='Optuna trials for xgb runs (0 = untuned)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--folds', type=int, default=10)
    ap.add_argument('--threads', type=int, default=12,
                    help='cores per task (must match NCORES_PER_TASK in sweep.pbs)')
    ap.add_argument('--python', default=sys.executable,
                    help='python of the gvfa environment (absolute path)')
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--cache-dir', default='cache')
    ap.add_argument('--out-dir', default='nci/cmds')
    ap.add_argument('--log-dir', default='logs')
    ap.add_argument('--group-folder', nargs='*', default=[], metavar='GROUP=DIR',
                    help='add or override a group folder (testing)')
    args = ap.parse_args()

    folders = dict(GROUP_FOLDERS)
    for item in args.group_folder:
        g, d = item.split('=', 1)
        folders[g] = d
    for g in args.groups:
        if g not in folders:
            raise SystemExit(f'unknown group {g}; known: {", ".join(folders)}')

    results_dir = ROOT / args.results_dir
    cache_dir = ROOT / args.cache_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    py = shlex.quote(args.python)
    t = args.threads
    # PYTHONUNBUFFERED: a task killed by the OOM killer or the timeout still
    # leaves its log up to the point it died, instead of an empty file.
    env = (f'OMP_NUM_THREADS={t} MKL_NUM_THREADS={t} OPENBLAS_NUM_THREADS={t} '
           f'TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=1')

    def command(tr, te, group, log, extra):
        q = shlex.quote
        return (f'cd {q(str(ROOT))} && {env} {py} Run_experiment.py '
                f'-i {q(str(tr))} -p {q(str(te))} --group {q(group)} '
                f'--seed {args.seed} --folds {args.folds} --n-jobs {t} '
                f'--cache-dir {q(args.cache_dir)} --results-dir {q(args.results_dir)} '
                f'{extra} > {q(log)} 2>&1')

    stage1, stage2, stage3 = [], [], []
    n_total = n_done = 0
    for group in args.groups:
        log_dir = f'{args.log_dir}/{group}'
        (ROOT / log_dir).mkdir(parents=True, exist_ok=True)
        for tr, te in dataset_pairs(group, folders[group]):
            ds = short_dataset_name(tr)
            size = tr.stat().st_size                      # cost proxy for ordering

            if not prepared(cache_dir, tr.name, args.seed):
                stage1.append((size, command(tr, te, group, f'{log_dir}/{ds}__prepare.log',
                                             '--features rdkit --prepare-only')))
            for nf, dim in itertools.product(args.node_features, args.dims):
                if embedded(cache_dir, tr.name, nf, dim):
                    continue
                stage2.append((size * dim, command(
                    tr, te, group, f'{log_dir}/{ds}__embed_nf{nf}_dim{dim}.log',
                    f'--features gvfa --node-features {nf} --dim {dim} --prepare-only')))

            grid = [(f, nf, d) for f in args.features
                    for nf in args.node_features for d in args.dims]
            if args.rdkit:
                grid.append(('rdkit', 7, 0))
            for (feat, nf, dim), clf, bal in itertools.product(grid, args.clfs,
                                                               args.balance):
                tune = args.tune if clf == 'xgb' else 0
                tag = run_tag(ds, feat, nf, dim, clf, bal, tune, args.seed, args.folds)
                n_total += 1
                if finished(results_dir, group, tag):
                    n_done += 1
                    continue
                extra = f'--features {feat} --clf {clf}'
                if feat != 'rdkit':
                    extra += f' --node-features {nf} --dim {dim}'
                if bal:
                    extra += ' --balance'
                if tune:
                    extra += f' --tune {tune}'
                cost = size * max(dim, 100) * (args.tune if tune else 1)
                stage3.append((cost, command(tr, te, group, f'{log_dir}/{tag}.log', extra)))

    for fname, lines in (('stage1_prepare.txt', stage1), ('stage2_embed.txt', stage2),
                         ('stage3_runs.txt', stage3)):
        lines.sort(key=lambda x: -x[0])
        (out_dir / fname).write_text(''.join(c + '\n' for _, c in lines))
        print(f'{out_dir / fname}: {len(lines)} commands')
    print(f'experiments: {n_total} in the grid, {n_done} already finished, '
          f'{len(stage3)} to run')


if __name__ == '__main__':
    main()
