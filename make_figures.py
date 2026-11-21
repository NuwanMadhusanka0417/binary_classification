#!/usr/bin/env python
'''
make_figures.py -- redraw the paper's figures from the merged results,
in the layout of the original draft figures (vertical bars, models in
alphabetical order along x, algorithm legend underneath).

  figure0_active_fraction.png   Binned fraction_active_class
  figure1_ranks.png             Figure 1, mean(tot_rank), confidence 0.80
  figure2_eff_active.png        Figure 2, mean(eff1), active class
  figure3_eff_inactive.png      Figure 3, mean(eff1), inactive class

Slice plotted: 18 node features, class balancing ON, no Optuna tuning -- the
best-performing combination. 3 feature sets x 6 dimensions x 3 classifiers
= 54 models, plus the RDKit-descriptor baseline.

Model names follow the paper:
    dim<D>_<clf>                 GVFA hypervectors alone
    bundling_dim<D>_<clf>        RP(96 descriptors) + GVFA
    traddesc_96_dim<D>_<clf>     96 RDKit descriptors || GVFA
    0rdkit_nonconf               96 RDKit descriptors alone (baseline)

Usage:
  python make_figures.py [--results results/all_results_merged.csv]
                         [--outdir figures] [--eps 0.20]
'''
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

NF, BALANCE, TUNED = 18, True, False
EFF_FLOOR = 0.70                     # "individual CP efficiency >= 0.7"
BASE = '0rdkit_nonconf'

# colours taken from the draft figures
EFF_COLOUR = {'baseline': '#45DCE4', 'lr': '#EE2222',
              'rf': '#1AA05A', 'xgb': '#2222CC'}
RANK_COLOUR = {'better': '#0E6B52', 'lr': '#EE2222',
               'rf': '#2222CC', 'xgb': '#C5C8EE'}
HIST_COLOUR = '#C6C60A'

PRETTY = {'gvfa': 'dim{d}_{c}', 'bundle': 'bundling_dim{d}_{c}',
          'concat': 'traddesc_96_dim{d}_{c}'}


def model_name(row):
    if row.features == 'rdkit':
        return BASE
    return PRETTY[row.features].format(d=int(row.dim), c=row.clf)


def load(path, eps):
    d = pd.read_csv(path)
    d = d[np.isclose(d.eps, eps)].copy()
    keep = ((d.features == 'rdkit') |
            ((d.node_features == NF) & (d.balance == BALANCE) & (d.tuned == TUNED)))
    d = d[keep].copy()
    d['model'] = d.apply(model_name, axis=1)
    d['algo'] = np.where(d.features == 'rdkit', 'baseline', d.clf)
    return d


def style(ax, xlabel, ylabel):
    """Boxed frame, light grid, vertical category labels -- as in the draft."""
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis='both', color='#d0d0d0', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color('#808080')
        s.set_linewidth(0.8)
    ax.tick_params(axis='y', labelsize=9)


def bars(ax, names, values, colours):
    x = np.arange(len(names))
    ax.bar(x, values, color=colours, width=0.62, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=6.0)
    ax.set_xlim(-0.8, len(names) - 0.2)


def legend(fig, entries):
    handles = [Line2D([], [], marker='o', linestyle='none', markersize=7,
                      markerfacecolor=c, markeredgecolor=c, label=l)
               for l, c in entries]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               frameon=False, fontsize=9, title='algorithm',
               title_fontsize=9, bbox_to_anchor=(0.5, 0.005))


# --------------------------------------------------------------------------
def figure0(d, out):
    frac = (d.groupby('dataset')
             .apply(lambda g: (g.n_active_train.iloc[0] + g.n_active_test.iloc[0]) /
                              (g.n_train.iloc[0] + g.n_test.iloc[0]),
                    include_groups=False))
    step = 0.02
    edges = np.arange(0, np.ceil(frac.max() / step) * step + step, step)
    counts, _ = np.histogram(frac, bins=edges)

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    ax.bar(edges[:-1] + step / 2, counts, width=step * 0.55,
           color=HIST_COLOUR, zorder=3)
    style(ax, 'Binned fraction_active_class', 'Count')
    ax.set_xticks(edges)
    ax.set_xticklabels([f'{e:g}' for e in edges], rotation=90, fontsize=7.5)
    ax.set_xlim(edges[0] - step * 0.6, edges[-1] + step * 0.6)
    ax.set_ylim(0, counts.max() + 1)
    ax.set_yticks(np.arange(0, counts.max() + 2, 2))
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'{out}  ({len(frac)} datasets, median active fraction {frac.median():.3f})')


def figure1(d, out):
    """mean(tot_rank): rank per dataset, valid models first then efficiency."""
    base_ds = set(d[d.features == 'rdkit'].dataset.unique())
    sub = d[d.dataset.isin(base_ds)]
    piv_ok = sub.pivot_table(index='dataset', columns='model', values='ok')
    piv_eff = sub.pivot_table(index='dataset', columns='model', values='eff')
    score = (piv_ok.astype(float) + piv_eff)      # valid (1) dominates eff (<1)
    # ranking needs a complete grid: drop incomplete DATASETS, not models
    score = score.dropna(axis=0, how='any').dropna(axis=1, how='all')
    mr = score.rank(axis=1, ascending=False).mean()

    names = sorted(mr.index)                       # alphabetical, as in the draft
    values = [mr[n] for n in names]
    algos = sub.drop_duplicates('model').set_index('model').algo
    base_v = mr[BASE]
    # 28 of 55 models beat the baseline here, so colouring those green would
    # swamp the classifier colours. Keep the algorithm encoding and mark the
    # baseline with a dashed line instead: anything below it is better.
    colours = [RANK_COLOUR['better'] if n == BASE else RANK_COLOUR[algos[n]]
               for n in names]

    fig, ax = plt.subplots(figsize=(10.6, 7.8))
    bars(ax, names, values, colours)
    style(ax, 'features', 'mean(tot_rank)')
    ax.set_ylim(0, max(values) * 1.08)
    ax.axhline(base_v, color=RANK_COLOUR['better'], linestyle='--',
               linewidth=1.3, zorder=4)
    ax.annotate(f'baseline {base_v:.1f} - bars below this line are better',
                xy=(0.995, base_v), xycoords=('axes fraction', 'data'),
                xytext=(0, 4), textcoords='offset points',
                ha='right', va='bottom', fontsize=8,
                color=RANK_COLOUR['better'], fontweight='bold')
    legend(fig, [('nonconf (baseline)', RANK_COLOUR['better']),
                 ('lr', RANK_COLOUR['lr']), ('rf', RANK_COLOUR['rf']),
                 ('xgb', RANK_COLOUR['xgb'])])
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'{out}  ({score.shape[1]} models, {score.shape[0]} datasets, '
          f'{int((mr <= base_v).sum()) - 1} better than or equal to the baseline)')
    print(f'  CAPTION: Figure 1. Rank results for the number of valid models at '
          f'confidence level 0.8. Bars below the dashed line are better (lower) than or '
          f'equal to the baseline (0rdkit_nonconf at the very left). {score.shape[1]} models '
          f'ranked over the {score.shape[0]} datasets carrying every model and the baseline.')


def figure2_3(d, out, col, number, note_class):
    sub = d[d.eff_active >= EFF_FLOOR]
    g = sub.groupby('model').agg(value=(col, 'mean'),
                                 n=('dataset', 'nunique'))
    algos = d.drop_duplicates('model').set_index('model').algo
    names = sorted(g.index)
    values = [g.value[n] for n in names]
    colours = [EFF_COLOUR[algos[n]] for n in names]

    fig, ax = plt.subplots(figsize=(10.6, 7.8))
    bars(ax, names, values, colours)
    style(ax, 'features', 'mean(eff1)')
    ax.set_ylim(0, max(values) * 1.10)
    ax.set_yticks(np.arange(0, max(values) * 1.10, 0.1))
    legend(fig, [('0rdkit_nonconf', EFF_COLOUR['baseline']),
                 ('lr', EFF_COLOUR['lr']), ('rf', EFF_COLOUR['rf']),
                 ('xgb', EFF_COLOUR['xgb'])])
    fig.suptitle(f'Figure {number}.  Average CP efficiency for the {note_class} class at '
                 f'confidence level 0.8 for models with individual\nCP efficiency >= '
                 f'{EFF_FLOOR:.1f} for the active class. GVFA models span '
                 f'{int(g.n.drop(BASE, errors="ignore").max())} datasets; the baseline '
                 f'0rdkit_nonconf only {int(g.n.get(BASE, 0))}.',
                 fontsize=9.5, y=0.02, va='bottom')
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    top = g.value.idxmax()
    print(f'{out}  (best: {top} = {g.value.max():.3f})')
    print(f'  CAPTION: Figure {number}. Average CP efficiency for the {note_class} class '
          f'at confidence level 0.8 for models with individual CP efficiency >= '
          f'{EFF_FLOOR:.1f} for the active class. GVFA models span '
          f'{int(g.n.drop(BASE, errors="ignore").max())} datasets; the baseline '
          f'0rdkit_nonconf only {int(g.n.get(BASE, 0))}.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/all_results_merged.csv')
    ap.add_argument('--outdir', default='figures')
    ap.add_argument('--eps', type=float, default=0.20)
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    d = load(a.results, a.eps)
    print(f'{d.run_id.nunique()} runs, {d.model.nunique()} models, '
          f'{d.dataset.nunique()} datasets, confidence {1-a.eps:.2f}\n')

    figure0(d, out / 'figure0_active_fraction.png')
    figure1(d, out / 'figure1_ranks.png')
    figure2_3(d, out / 'figure2_eff_active.png', 'eff_active', 2, 'active')
    figure2_3(d, out / 'figure3_eff_inactive.png', 'eff_inactive', 3, 'inactive')


if __name__ == '__main__':
    main()
