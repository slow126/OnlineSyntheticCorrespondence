#!/usr/bin/env python3
"""Generate tab_sampling_stability.tex from the sharded density-invariance data.
Mirrors blocks.py R2A_METRICS: Spearman of each metric at budget N vs its value
at the largest budget, over the train->eval pairs."""
import pandas as pd, numpy as np
from pathlib import Path

s = pd.read_csv('analysis_v3/density_invariance_pair_sharded/'
                'stability_flow_train_eval__eval_eval.csv')
s = s[s.pair_type == 'train_eval']

# (metric col, printed label) in blocks.py R2A order
ROWS = [
    ('mean_nn_b_to_a',       r'mean-NN coverage $\dbt$'),
    ('mean_nn_a_to_b',       r'mean-NN off-target $\dtb$'),
    ('mean_nn_sym',          r'mean-NN symmetric'),
    ('b_covered_by_a_eps4px', r'$\epsilon$-coverage 4\,px $\dbt$'),
    ('a_covered_by_b_eps4px', r'$\epsilon$-coverage 4\,px $\dtb$'),
    ('kl_b_to_a_k20',        r'kNN-KL ($k{=}20$) $\dbt$'),
    ('kl_a_to_b_k20',        r'kNN-KL ($k{=}20$) $\dtb$'),
]
piv = s.pivot_table(index='metric', columns='level', values='rho')
budgets = sorted(piv.columns)

def hdr(n):
    if n >= 1_000_000: return f'{n//1_000_000}M'
    return f'{n//1000}k'

lines = []
lines.append(r'\begin{tabular}{l ' + 'r'*len(budgets) + '}')
lines.append(r'\toprule')
lines.append('Estimator & ' + ' & '.join(hdr(b) for b in budgets) + r' \\')
lines.append(r'\cmidrule(lr){1-1}\cmidrule(lr){2-' + str(1+len(budgets)) + '}')
lines.append(r'\multicolumn{' + str(1+len(budgets)) +
             r'}{@{}l}{\textit{support geometry (used)}}\\')
seen_kl = False
for col, label in ROWS:
    if col.startswith('kl') and not seen_kl:
        lines.append(r'\midrule')
        lines.append(r'\multicolumn{' + str(1+len(budgets)) +
                     r'}{@{}l}{\textit{density (kNN-KL, discarded)}}\\')
        seen_kl = True
    vals = [piv.loc[col, b] for b in budgets]
    cells = ' & '.join(f'{v:.3f}' for v in vals)
    lines.append(f'{label} & {cells} ' + r'\\')
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
tex = '\n'.join(lines)

out = Path('ACCV_2026/tables/tab_sampling_stability.tex')
out.write_text(tex + '\n')
print('wrote', out)
print(tex)
print('\nn_pairs per level:', dict(s.groupby("level").n_pairs.max()))
