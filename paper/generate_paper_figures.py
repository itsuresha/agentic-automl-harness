#!/usr/bin/env python3
"""Generate all publication-quality figures and tables for the Harness paper."""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path("/mnt/user-data/uploads/AIDE/cp3_fault_injection")
OUT  = Path("/home/claude/paper_figures")
OUT.mkdir(exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'gate':    '#2563EB',  # blue
    'monitor': '#059669',  # green
    'oracle':  '#6B7280',  # gray
    'F1':      '#EF4444',  # red
    'F2':      '#F59E0B',  # amber
    'F3':      '#8B5CF6',  # purple
    'clean':   '#10B981',  # emerald
    'dev':     '#3B82F6',  # blue
    'cp6':     '#F97316',  # orange
    'crc':     '#2563EB',
    'naive':   '#DC2626',
}

# ── Load Data ────────────────────────────────────────────────────────────────
cp5_eval = json.loads((BASE / "cp5_results/cp5_evaluation.json").read_text())
cp6_eval = json.loads((BASE / "cp6_results/cp6_evaluation.json").read_text())
oracle   = json.loads((BASE / "oracle/oracle_evaluation.json").read_text())
cp3_sum  = json.loads((BASE / "cp3_summary.json").read_text())

conformal_thresh = pd.read_csv(BASE / "cp5_results/conformal_thresholds.csv")
lodo_cv          = pd.read_csv(BASE / "cp5_results/lodo_cross_conformal.csv")
per_class_dev    = pd.read_csv(BASE / "cp5_results/per_class_analysis.csv")
gated_dev        = pd.read_csv(BASE / "cp5_results/gated_decisions.csv")
dev_features     = pd.read_csv(BASE / "features/step_features.csv")
cp6_features     = pd.read_csv(BASE / "cp6_results/cp6_features.csv")
cp6_labels       = pd.read_csv(BASE / "cp6_results/cp6_labels.csv")
cp6_per_ds       = pd.read_csv(BASE / "cp6_results/cp6_per_dataset.csv")
cp6_per_class    = pd.read_csv(BASE / "cp6_results/cp6_per_class.csv")
cp6_det_comp     = pd.read_csv(BASE / "cp6_results/cp6_detector_comparison.csv")
cp6_gate_pred    = pd.read_csv(BASE / "cp6_results/cp6_gate_predictions.csv")
cp6_mon_pred     = pd.read_csv(BASE / "cp6_results/cp6_monitor_predictions.csv")
combined_labels  = pd.read_csv(BASE / "labels/combined_labels.csv")
cp6_trace        = pd.read_csv(BASE / "cp6_results/cp6_trace_summary.csv")
f2_deltas        = pd.read_csv(BASE / "analysis/f2_score_deltas.csv")
f3_deltas        = pd.read_csv(BASE / "analysis/f3_score_deltas.csv")

print("All data loaded successfully.")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1: FSR vs α (CRC threshold sweep) — Dev corpus
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 3.8))
alphas = conformal_thresh['alpha']
ax.plot(alphas, conformal_thresh['crc_fsr'], 'o-', color=COLORS['crc'],
        label='CRC threshold', linewidth=1.8, markersize=5)
ax.plot(alphas, conformal_thresh['naive_fsr'], 's--', color=COLORS['naive'],
        label='Naïve quantile', linewidth=1.4, markersize=4)
ax.plot([0, 0.35], [0, 0.35], ':', color='#9CA3AF', linewidth=1, label='Ideal (FSR = α)')
ax.set_xlabel('Target false-skip rate α')
ax.set_ylabel('Empirical false-skip rate (FSR)')
ax.set_title('CRC Threshold Calibration (Dev Corpus, n = 208)')
ax.legend(framealpha=0.9)
ax.set_xlim(-0.01, 0.32)
ax.set_ylim(-0.01, 0.32)
fig.savefig(OUT / "fig1_fsr_vs_alpha.pdf")
fig.savefig(OUT / "fig1_fsr_vs_alpha.png")
plt.close(fig)
print("✓ Figure 1: FSR vs α")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2: ROC curves — Gate and Monitor on CP-6
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

# Gate ROC
fpr_g, tpr_g, _ = roc_curve(cp6_gate_pred['y_true'], cp6_gate_pred['y_prob'])
roc_auc_g = auc(fpr_g, tpr_g)
axes[0].plot(fpr_g, tpr_g, color=COLORS['gate'], linewidth=2,
             label=f'Gate AUC = {roc_auc_g:.3f}')
axes[0].plot([0, 1], [0, 1], ':', color='#9CA3AF', linewidth=1)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('(a) Gate Tier — ROC')
axes[0].legend(loc='lower right')

# Monitor ROC
fpr_m, tpr_m, _ = roc_curve(cp6_mon_pred['y_true'], cp6_mon_pred['y_prob'])
roc_auc_m = auc(fpr_m, tpr_m)
axes[1].plot(fpr_m, tpr_m, color=COLORS['monitor'], linewidth=2,
             label=f'Monitor AUC = {roc_auc_m:.3f}')
axes[1].plot([0, 1], [0, 1], ':', color='#9CA3AF', linewidth=1)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('(b) Monitor Tier — ROC')
axes[1].legend(loc='lower right')

fig.tight_layout()
fig.savefig(OUT / "fig2_roc_curves.pdf")
fig.savefig(OUT / "fig2_roc_curves.png")
plt.close(fig)
print("✓ Figure 2: ROC curves")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Precision-Recall curves — Gate and Monitor on CP-6
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

prec_g, rec_g, _ = precision_recall_curve(cp6_gate_pred['y_true'], cp6_gate_pred['y_prob'])
pr_auc_g = auc(rec_g, prec_g)
axes[0].plot(rec_g, prec_g, color=COLORS['gate'], linewidth=2,
             label=f'Gate AP = {pr_auc_g:.3f}')
no_skill_g = cp6_gate_pred['y_true'].mean()
axes[0].axhline(y=no_skill_g, color='#9CA3AF', linestyle=':', linewidth=1, label=f'No skill ({no_skill_g:.2f})')
axes[0].set_xlabel('Recall')
axes[0].set_ylabel('Precision')
axes[0].set_title('(a) Gate Tier — PR')
axes[0].legend(loc='lower left')
axes[0].set_xlim(-0.02, 1.02)
axes[0].set_ylim(0, 1.05)

prec_m, rec_m, _ = precision_recall_curve(cp6_mon_pred['y_true'], cp6_mon_pred['y_prob'])
pr_auc_m = auc(rec_m, prec_m)
axes[1].plot(rec_m, prec_m, color=COLORS['monitor'], linewidth=2,
             label=f'Monitor AP = {pr_auc_m:.3f}')
axes[1].axhline(y=no_skill_g, color='#9CA3AF', linestyle=':', linewidth=1, label=f'No skill ({no_skill_g:.2f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('(b) Monitor Tier — PR')
axes[1].legend(loc='lower left')
axes[1].set_xlim(-0.02, 1.02)
axes[1].set_ylim(0, 1.05)

fig.tight_layout()
fig.savefig(OUT / "fig3_pr_curves.pdf")
fig.savefig(OUT / "fig3_pr_curves.png")
plt.close(fig)
print("✓ Figure 3: PR curves")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Per-dataset gate FSR bar chart — CP-6
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 3.5))
ds_order = cp6_per_ds.sort_values('gate_fsr', ascending=False)['dataset'].tolist()
bars = ax.barh(ds_order, cp6_per_ds.set_index('dataset').loc[ds_order, 'gate_fsr'],
               color=COLORS['gate'], alpha=0.8, height=0.6)
ax.axvline(x=0.10, color=COLORS['naive'], linestyle='--', linewidth=1.5, label='α = 0.10')
ax.axvline(x=0.20, color='#9CA3AF', linestyle=':', linewidth=1, label='Per-dataset ≤ 0.20')
ax.set_xlabel('Gate FSR')
ax.set_title('Per-Dataset Gate FSR (CP-6 External Validation)')
ax.legend(loc='lower right', fontsize=8)
# Add value labels
for bar, ds in zip(bars, ds_order):
    v = cp6_per_ds.set_index('dataset').loc[ds, 'gate_fsr']
    ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
            f'{v:.2f}', va='center', fontsize=8)
ax.set_xlim(0, 0.25)
fig.tight_layout()
fig.savefig(OUT / "fig4_per_dataset_fsr.pdf")
fig.savefig(OUT / "fig4_per_dataset_fsr.png")
plt.close(fig)
print("✓ Figure 4: Per-dataset FSR")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Confusion Matrices — Gate and Monitor on CP-6
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

for i, (tier, data) in enumerate([
    ('Gate', cp6_eval['crc_evaluation']['gate']),
    ('Monitor', cp6_eval['crc_evaluation']['monitor'])
]):
    cm = np.array([[data['tn'], data['fp']], [data['fn'], data['tp']]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Clean', 'Faulty'], yticklabels=['Clean', 'Faulty'],
                annot_kws={'size': 14}, cbar=False, linewidths=0.5)
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')
    axes[i].set_title(f'({chr(97+i)}) {tier} Tier')

fig.suptitle('Confusion Matrices — CP-6 External Validation (n = 89)', y=1.02, fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig5_confusion_matrices.pdf")
fig.savefig(OUT / "fig5_confusion_matrices.png")
plt.close(fig)
print("✓ Figure 5: Confusion matrices")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Per-class recall — Gate vs Monitor on CP-6
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5, 3.5))
classes = ['F1 (RCP)', 'F2 (DL)', 'F3 (TBF)']
gate_recall = [0.9394, 1.0, 0.7778]
mon_recall  = [0.9091, 1.0, 0.7222]
x = np.arange(len(classes))
w = 0.32
b1 = ax.bar(x - w/2, gate_recall, w, label='Gate', color=COLORS['gate'], alpha=0.85)
b2 = ax.bar(x + w/2, mon_recall,  w, label='Monitor', color=COLORS['monitor'], alpha=0.85)
ax.set_ylabel('Recall')
ax.set_title('Per-Class Recall on CP-6 (n = 89)')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim(0, 1.12)
ax.legend()
# Value labels
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f'{h:.2f}',
                ha='center', va='bottom', fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig6_per_class_recall.pdf")
fig.savefig(OUT / "fig6_per_class_recall.png")
plt.close(fig)
print("✓ Figure 6: Per-class recall")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Detector comparison — P/R/F1 grouped bars for CP-6
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4))
det_names = cp6_det_comp['detector'].tolist()
det_labels = [f"{d}\n→{t}" for d, t in zip(det_names, cp6_det_comp['target'])]
x = np.arange(len(det_names))
w = 0.25
ax.bar(x - w, cp6_det_comp['precision'], w, label='Precision', color='#3B82F6', alpha=0.85)
ax.bar(x,     cp6_det_comp['recall'],    w, label='Recall',    color='#10B981', alpha=0.85)
ax.bar(x + w, cp6_det_comp['f1'],        w, label='F1',        color='#F59E0B', alpha=0.85)
ax.set_ylabel('Score')
ax.set_title('Individual Detector Performance (CP-6)')
ax.set_xticks(x)
ax.set_xticklabels(det_labels, fontsize=7.5)
ax.set_ylim(0, 1.12)
ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig(OUT / "fig7_detector_comparison.pdf")
fig.savefig(OUT / "fig7_detector_comparison.png")
plt.close(fig)
print("✓ Figure 7: Detector comparison")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 8: Cost savings waterfall — CP-6
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 3.8))
savings = cp6_eval['tier1_savings']
categories = ['Oracle\nceiling', 'Gross\nsavings', 'FP cost', 'Net\nsavings']
values = [savings['oracle_ceiling_pct'], savings['gross_savings_pct'],
          -savings['fp_cost_pct'], savings['net_savings_pct']]
colors_w = ['#6B7280', '#3B82F6', '#EF4444', '#10B981']
bars = ax.bar(categories, values, color=colors_w, alpha=0.85, width=0.55)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1 if v >= 0 else bar.get_height() - 3,
            f'{v:+.1f}%', ha='center', va='bottom' if v >= 0 else 'top', fontsize=9, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylabel('Wall-clock time (%)')
ax.set_title('Tier-1 Gate Cost Savings (CP-6)')
ax.set_ylim(-20, 65)
fig.tight_layout()
fig.savefig(OUT / "fig8_cost_savings.pdf")
fig.savefig(OUT / "fig8_cost_savings.png")
plt.close(fig)
print("✓ Figure 8: Cost savings")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 9: LODO coverage heatmap — Dev corpus
# ═════════════════════════════════════════════════════════════════════════════
alphas_lodo = sorted(lodo_cv['alpha'].unique())
datasets_lodo = sorted(lodo_cv['dataset'].unique())
pivot = lodo_cv.pivot_table(index='dataset', columns='alpha', values='covered', aggfunc='first')
pivot = pivot.reindex(columns=alphas_lodo).astype(float)

fig, ax = plt.subplots(figsize=(5.5, 6))
sns.heatmap(pivot, annot=False, cmap=['#FCA5A5', '#86EFAC'], ax=ax,
            cbar=False, linewidths=0.5, linecolor='white',
            xticklabels=[str(a) for a in alphas_lodo])
# Add check / × marks
for i, ds in enumerate(pivot.index):
    for j, alpha in enumerate(pivot.columns):
        v = pivot.loc[ds, alpha]
        ax.text(j + 0.5, i + 0.5, 'Y' if v == 1 else 'N',
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='#166534' if v == 1 else '#991B1B')
ax.set_xlabel('α')
ax.set_ylabel('')
ax.set_title('LODO Coverage (FSR ≤ α per held-out dataset)')

# Add coverage summary at bottom
coverage_summary = cp5_eval['lodo_coverage_summary']
summary_text = ' | '.join([f"α={a}: {coverage_summary[a]['n_covered']}/16"
                           for a in ['0.01', '0.05', '0.1', '0.15', '0.2', '0.3']])
ax.text(0.5, -0.06, summary_text, ha='center', va='top', transform=ax.transAxes, fontsize=7.5)

fig.tight_layout()
fig.savefig(OUT / "fig9_lodo_coverage.pdf")
fig.savefig(OUT / "fig9_lodo_coverage.png")
plt.close(fig)
print("✓ Figure 9: LODO coverage")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 10: Score probability distributions — Gate predictions on CP-6
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 3.5))
faulty_probs = cp6_gate_pred[cp6_gate_pred['y_true'] == 1]['y_prob']
clean_probs  = cp6_gate_pred[cp6_gate_pred['y_true'] == 0]['y_prob']
ax.hist(clean_probs, bins=20, alpha=0.6, color=COLORS['clean'], label='Clean steps', density=True)
ax.hist(faulty_probs, bins=20, alpha=0.6, color=COLORS['F1'], label='Faulty steps', density=True)
ax.axvline(x=0.324949, color=COLORS['gate'], linestyle='--', linewidth=2,
           label=f'λ_gate = 0.325')
ax.set_xlabel('Predicted fault probability')
ax.set_ylabel('Density')
ax.set_title('Gate Score Distributions (CP-6, n = 89)')
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig10_score_distributions.pdf")
fig.savefig(OUT / "fig10_score_distributions.png")
plt.close(fig)
print("✓ Figure 10: Score distributions")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 11: F2 and F3 injection effect sizes
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

# F2: score inflation
axes[0].hist(f2_deltas['delta'], bins=15, color=COLORS['F2'], alpha=0.75, edgecolor='white')
axes[0].axvline(x=f2_deltas['delta'].mean(), color='black', linestyle='--', linewidth=1.5,
                label=f"Mean = {f2_deltas['delta'].mean():.4f}")
axes[0].axvline(x=0, color='#9CA3AF', linestyle=':', linewidth=1)
axes[0].set_xlabel('Score delta (injected − original)')
axes[0].set_ylabel('Count')
axes[0].set_title('(a) F2 Data Leakage: Score Inflation')
axes[0].legend(fontsize=8)

# F3: score degradation
axes[1].hist(f3_deltas['delta'], bins=20, color=COLORS['F3'], alpha=0.75, edgecolor='white')
axes[1].axvline(x=f3_deltas['delta'].mean(), color='black', linestyle='--', linewidth=1.5,
                label=f"Mean = {f3_deltas['delta'].mean():.4f}")
axes[1].axvline(x=0, color='#9CA3AF', linestyle=':', linewidth=1)
axes[1].set_xlabel('Score delta (injected − original)')
axes[1].set_ylabel('Count')
axes[1].set_title('(b) F3 Training-Budget Fault: Score Degradation')
axes[1].legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT / "fig11_injection_effects.pdf")
fig.savefig(OUT / "fig11_injection_effects.png")
plt.close(fig)
print("✓ Figure 11: Injection effect sizes")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 12: Per-dataset net savings — CP-6
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 3.5))
ds_sorted = cp6_per_ds.sort_values('net_savings_pct', ascending=True)
colors_bar = [COLORS['gate'] if v >= 0 else '#EF4444' for v in ds_sorted['net_savings_pct']]
bars = ax.barh(ds_sorted['dataset'], ds_sorted['net_savings_pct'],
               color=colors_bar, alpha=0.85, height=0.55)
ax.axvline(x=0, color='black', linewidth=0.5)
ax.axvline(x=39.64, color=COLORS['monitor'], linestyle='--', linewidth=1.2,
           label=f'Aggregate net = 39.6%')
for bar, v in zip(bars, ds_sorted['net_savings_pct']):
    ax.text(v + (2 if v >= 0 else -2), bar.get_y() + bar.get_height()/2,
            f'{v:.1f}%', va='center', ha='left' if v >= 0 else 'right', fontsize=8)
ax.set_xlabel('Net wall-clock savings (%)')
ax.set_title('Per-Dataset Cost Savings (CP-6)')
ax.legend(loc='lower right', fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig12_per_dataset_savings.pdf")
fig.savefig(OUT / "fig12_per_dataset_savings.png")
plt.close(fig)
print("✓ Figure 12: Per-dataset savings")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 13: Dev vs CP-6 — Recall comparison by fault class
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5, 3.5))
classes = ['F1', 'F2', 'F3', 'Any']
# Dev gate recall from cp5_evaluation: per_class_alpha_0.10
dev_recall_gate = [1.0, 1.0, 0.725, cp5_eval['key_alpha_sweep'][2]['crc_recall']]  # α=0.10
cp6_recall_gate = [0.9394, 1.0, 0.7778, 0.9259]
x = np.arange(len(classes))
w = 0.32
b1 = ax.bar(x - w/2, dev_recall_gate, w, label='Dev (n=208)', color=COLORS['dev'], alpha=0.85)
b2 = ax.bar(x + w/2, cp6_recall_gate, w, label='CP-6 (n=89)', color=COLORS['cp6'], alpha=0.85)
ax.set_ylabel('Gate Recall')
ax.set_title('Internal vs External Validation — Gate Recall')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim(0, 1.15)
ax.legend()
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f'{h:.2f}',
                ha='center', va='bottom', fontsize=7.5)
fig.tight_layout()
fig.savefig(OUT / "fig13_dev_vs_cp6_recall.pdf")
fig.savefig(OUT / "fig13_dev_vs_cp6_recall.png")
plt.close(fig)
print("✓ Figure 13: Dev vs CP-6 recall")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 14: Fault class distribution — Dev and CP-6
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

# Dev
dev_f1 = combined_labels['F1'].sum()
dev_f2 = combined_labels['F2'].sum()
dev_f3 = combined_labels['F3'].sum()
dev_clean = len(combined_labels) - combined_labels['any_fault'].sum()
labels_dev = ['F1 (RCP)', 'F2 (DL)', 'F3 (TBF)', 'Clean']
sizes_dev = [dev_f1, dev_f2, dev_f3, dev_clean]
colors_pie = [COLORS['F1'], COLORS['F2'], COLORS['F3'], COLORS['clean']]
axes[0].pie(sizes_dev, labels=labels_dev, colors=colors_pie, autopct='%1.0f%%',
            startangle=90, textprops={'fontsize': 8})
axes[0].set_title(f'(a) Dev Corpus (n = {len(combined_labels)})')

# CP-6
cp6_f1 = cp6_labels['F1'].sum()
cp6_f2 = cp6_labels['F2'].sum()
cp6_f3 = cp6_labels['F3'].sum()
cp6_clean = len(cp6_labels) - cp6_labels['any_fault'].sum()
sizes_cp6 = [cp6_f1, cp6_f2, cp6_f3, cp6_clean]
axes[1].pie(sizes_cp6, labels=labels_dev, colors=colors_pie, autopct='%1.0f%%',
            startangle=90, textprops={'fontsize': 8})
axes[1].set_title(f'(b) CP-6 Corpus (n = {len(cp6_labels)})')

fig.suptitle('Fault Class Distribution', y=1.02, fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig14_fault_distribution.pdf")
fig.savefig(OUT / "fig14_fault_distribution.png")
plt.close(fig)
print("✓ Figure 14: Fault distribution")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 15: LODO FSR distribution at α = 0.10 — Dev corpus
# ═════════════════════════════════════════════════════════════════════════════
lodo_10 = lodo_cv[lodo_cv['alpha'] == 0.10].copy()
lodo_10 = lodo_10.sort_values('test_fsr', ascending=False)

fig, ax = plt.subplots(figsize=(6, 4))
colors_lodo = [COLORS['gate'] if fsr <= 0.10 else '#EF4444' for fsr in lodo_10['test_fsr']]
bars = ax.barh(lodo_10['dataset'], lodo_10['test_fsr'], color=colors_lodo, alpha=0.8, height=0.6)
ax.axvline(x=0.10, color=COLORS['naive'], linestyle='--', linewidth=1.5, label='α = 0.10')
ax.set_xlabel('Test FSR')
ax.set_title('LODO Cross-Conformal FSR at α = 0.10')
ax.legend(loc='lower right', fontsize=8)
for bar, fsr in zip(bars, lodo_10['test_fsr']):
    ax.text(fsr + 0.005, bar.get_y() + bar.get_height()/2,
            f'{fsr:.2f}', va='center', fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig15_lodo_fsr_at_010.pdf")
fig.savefig(OUT / "fig15_lodo_fsr_at_010.png")
plt.close(fig)
print("✓ Figure 15: LODO FSR at α=0.10")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 16: CP-6 API cost by dataset
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 3.5))
trace_sorted = cp6_trace.sort_values('api_cost_usd', ascending=True)
ax.barh(trace_sorted['dataset_name'], trace_sorted['api_cost_usd'],
        color=COLORS['cp6'], alpha=0.85, height=0.55)
total_cost = cp6_trace['api_cost_usd'].sum()
ax.axvline(x=cp6_trace['api_cost_usd'].mean(), color='black', linestyle='--',
           linewidth=1.2, label=f'Mean = ${cp6_trace["api_cost_usd"].mean():.2f}')
for i, (_, row) in enumerate(trace_sorted.iterrows()):
    ax.text(row['api_cost_usd'] + 0.03, i, f'${row["api_cost_usd"]:.2f}',
            va='center', fontsize=8)
ax.set_xlabel('API cost (USD)')
ax.set_title(f'CP-6 API Cost per Dataset (Total = ${total_cost:.2f})')
ax.legend(loc='lower right', fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig16_cp6_api_cost.pdf")
fig.savefig(OUT / "fig16_cp6_api_cost.png")
plt.close(fig)
print("✓ Figure 16: CP-6 API cost")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 17: F1/Recall/F1 comparison — Dev (CP-5R) vs CP-6 aggregate
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 3.8))
metrics = ['FSR', 'Recall', 'Precision', 'F1']
# Dev gate at α=0.10
dev_vals = [
    cp5_eval['key_alpha_sweep'][2]['crc_fsr'],
    cp5_eval['key_alpha_sweep'][2]['crc_recall'],
    cp5_eval['key_alpha_sweep'][2]['crc_precision'],
    cp5_eval['key_alpha_sweep'][2]['crc_f1']
]
cp6_vals = [
    cp6_eval['crc_evaluation']['gate']['fsr'],
    cp6_eval['crc_evaluation']['gate']['recall'],
    cp6_eval['crc_evaluation']['gate']['precision'],
    cp6_eval['crc_evaluation']['gate']['f1']
]
x = np.arange(len(metrics))
w = 0.32
b1 = ax.bar(x - w/2, dev_vals, w, label='Dev (n=208)', color=COLORS['dev'], alpha=0.85)
b2 = ax.bar(x + w/2, cp6_vals, w, label='CP-6 (n=89)', color=COLORS['cp6'], alpha=0.85)
ax.set_ylabel('Score')
ax.set_title('Gate Metrics: Internal vs External Validation')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.15)
ax.legend()
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f'{h:.3f}',
                ha='center', va='bottom', fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig17_dev_vs_cp6_gate_metrics.pdf")
fig.savefig(OUT / "fig17_dev_vs_cp6_gate_metrics.png")
plt.close(fig)
print("✓ Figure 17: Dev vs CP-6 gate metrics")


# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
# TABLES (LaTeX format)
# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════

tables_dir = OUT / "tables"
tables_dir.mkdir(exist_ok=True)

# ── TABLE 1: Dataset Summary ─────────────────────────────────────────────────
# Dev corpus datasets
dev_datasets = combined_labels.groupby('dataset').agg(
    n_steps=('any_fault', 'count'),
    n_faulty=('any_fault', 'sum'),
    n_F1=('F1', 'sum'),
    n_F2=('F2', 'sum'),
    n_F3=('F3', 'sum'),
).reset_index()
dev_datasets['n_faulty'] = dev_datasets['n_faulty'].astype(int)
dev_datasets['n_F1'] = dev_datasets['n_F1'].astype(int)
dev_datasets['n_F2'] = dev_datasets['n_F2'].astype(int)
dev_datasets['n_F3'] = dev_datasets['n_F3'].astype(int)
dev_datasets['corpus'] = 'Dev'

# CP-6 datasets
cp6_datasets = cp6_labels.groupby('dataset').agg(
    n_steps=('any_fault', 'count'),
    n_faulty=('any_fault', 'sum'),
    n_F1=('F1', 'sum'),
    n_F2=('F2', 'sum'),
    n_F3=('F3', 'sum'),
).reset_index()
cp6_datasets['n_faulty'] = cp6_datasets['n_faulty'].astype(int)
cp6_datasets['n_F1'] = cp6_datasets['n_F1'].astype(int)
cp6_datasets['n_F2'] = cp6_datasets['n_F2'].astype(int)
cp6_datasets['n_F3'] = cp6_datasets['n_F3'].astype(int)
cp6_datasets['corpus'] = 'CP-6'

all_datasets = pd.concat([dev_datasets, cp6_datasets], ignore_index=True)
all_datasets = all_datasets.sort_values(['corpus', 'dataset'])

lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{Dataset summary for the development (Dev) and external validation (CP-6) corpora. Each corpus is derived from 20-step AIDE runs on OpenML CC-18 datasets.}")
lines.append(r"\label{tab:dataset_summary}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{llrrrrr}")
lines.append(r"\toprule")
lines.append(r"Corpus & Dataset & Steps & Faulty & F1 & F2 & F3 \\")
lines.append(r"\midrule")

for corpus in ['Dev', 'CP-6']:
    subset = all_datasets[all_datasets['corpus'] == corpus]
    for _, row in subset.iterrows():
        lines.append(f"{row['corpus']} & {row['dataset']} & {row['n_steps']} & {row['n_faulty']} & {row['n_F1']} & {row['n_F2']} & {row['n_F3']} \\\\")
    # Add subtotal
    tot = subset[['n_steps', 'n_faulty', 'n_F1', 'n_F2', 'n_F3']].sum()
    lines.append(r"\cmidrule{2-7}")
    lines.append(f"& \\textbf{{Total}} & \\textbf{{{int(tot['n_steps'])}}} & \\textbf{{{int(tot['n_faulty'])}}} & \\textbf{{{int(tot['n_F1'])}}} & \\textbf{{{int(tot['n_F2'])}}} & \\textbf{{{int(tot['n_F3'])}}} \\\\")
    if corpus == 'Dev':
        lines.append(r"\midrule")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

(tables_dir / "table1_dataset_summary.tex").write_text('\n'.join(lines))
print("✓ Table 1: Dataset summary")

# ── TABLE 2: CRC Threshold Sweep ─────────────────────────────────────────────
lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{CRC threshold calibration across target false-skip rates. The operating point $\alpha = 0.10$ (highlighted) is used for external validation.}")
lines.append(r"\label{tab:crc_thresholds}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{rrrrrrr}")
lines.append(r"\toprule")
lines.append(r"$\alpha$ & $\hat{\lambda}$ & FSR & Recall & Prec. & F1 & Flagged \\")
lines.append(r"\midrule")
for _, row in conformal_thresh.iterrows():
    prefix = r"\rowcolor{blue!8} " if row['alpha'] == 0.10 else ""
    lines.append(f"{prefix}{row['alpha']:.2f} & {row['crc_threshold']:.4f} & {row['crc_fsr']:.4f} & {row['crc_recall']:.4f} & {row['crc_precision']:.4f} & {row['crc_f1']:.4f} & {int(row['crc_n_flagged'])} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
(tables_dir / "table2_crc_thresholds.tex").write_text('\n'.join(lines))
print("✓ Table 2: CRC thresholds")

# ── TABLE 3: CP-6 Acceptance Criteria ─────────────────────────────────────────
lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{CP-6 acceptance criteria (30/32 passed). Required criteria are marked with~$^\dagger$.}")
lines.append(r"\label{tab:cp6_criteria}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{lllr}")
lines.append(r"\toprule")
lines.append(r"Criterion & Threshold & Result & Status \\")
lines.append(r"\midrule")

criteria = [
    ("Gate FSR$^\dagger$", r"$\leq$ 0.10", f"{cp6_eval['crc_evaluation']['gate']['fsr']:.4f}", "PASS" if cp6_eval['success_criteria']['gate_fsr_le_010'] else "FAIL"),
    ("Monitor FSR$^\dagger$", r"$\leq$ 0.10", f"{cp6_eval['crc_evaluation']['monitor']['fsr']:.4f}", "PASS" if cp6_eval['success_criteria']['monitor_fsr_le_010'] else "FAIL"),
    ("Gate F1$^\dagger$", r"$\geq$ 0.70", f"{cp6_eval['crc_evaluation']['gate']['f1']:.4f}", "PASS" if cp6_eval['success_criteria']['gate_f1_ge_070'] else "FAIL"),
    ("Net savings$^\dagger$", r"$>$ 0\%", f"{cp6_eval['tier1_savings']['net_savings_pct']:.2f}\\%", "PASS" if cp6_eval['success_criteria']['tier1_net_savings_gt_0'] else "FAIL"),
    ("Per-ds FSR$\leq$0.20", "6/6 datasets", f"{cp6_eval['n_gate_covered_020']}/6", "PASS" if cp6_eval['success_criteria']['stretch_gate_coverage_020'] else "FAIL"),
]

for name, thresh, result, status in criteria:
    status_fmt = r"\textcolor{green!60!black}{\checkmark}" if status == "PASS" else r"\textcolor{red}{\texttimes}"
    lines.append(f"{name} & {thresh} & {result} & {status_fmt} \\\\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
(tables_dir / "table3_cp6_criteria.tex").write_text('\n'.join(lines))
print("✓ Table 3: CP-6 acceptance criteria")

# ── TABLE 4: Per-Dataset Results — CP-6 ──────────────────────────────────────
lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{Per-dataset results on the CP-6 external validation corpus.}")
lines.append(r"\label{tab:cp6_per_dataset}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{lrrrrrr}")
lines.append(r"\toprule")
lines.append(r"Dataset & Steps & Faulty & Gate FSR & TP & FP & Net \% \\")
lines.append(r"\midrule")
for _, row in cp6_per_ds.sort_values('dataset').iterrows():
    ns = f"{row['net_savings_pct']:.1f}"
    lines.append(f"{row['dataset']} & {int(row['n_steps'])} & {int(row['n_faulty'])} & {row['gate_fsr']:.4f} & {int(row['gate_tp'])} & {int(row['gate_fp'])} & {ns} \\\\")
lines.append(r"\midrule")
# Aggregate
lines.append(f"\\textbf{{Aggregate}} & \\textbf{{89}} & \\textbf{{54}} & \\textbf{{{cp6_eval['crc_evaluation']['gate']['fsr']:.4f}}} & \\textbf{{{cp6_eval['crc_evaluation']['gate']['tp']}}} & \\textbf{{{cp6_eval['crc_evaluation']['gate']['fp']}}} & \\textbf{{{cp6_eval['tier1_savings']['net_savings_pct']:.1f}}} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
(tables_dir / "table4_cp6_per_dataset.tex").write_text('\n'.join(lines))
print("✓ Table 4: CP-6 per-dataset results")

# ── TABLE 5: Detector Comparison — CP-6 ──────────────────────────────────────
lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{Individual detector performance on the CP-6 corpus. D1 and D2 are rule-based; D3, D\_any are LightGBM.}")
lines.append(r"\label{tab:detector_comparison}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{llrrrrrr}")
lines.append(r"\toprule")
lines.append(r"Detector & Target & Prec. & Recall & F1 & TP & FP & FN \\")
lines.append(r"\midrule")
for _, row in cp6_det_comp.iterrows():
    lines.append(f"{row['detector']} & {row['target']} & {row['precision']:.4f} & {row['recall']:.4f} & {row['f1']:.4f} & {int(row['tp'])} & {int(row['fp'])} & {int(row['fn'])} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
(tables_dir / "table5_detector_comparison.tex").write_text('\n'.join(lines))
print("✓ Table 5: Detector comparison")

# ── TABLE 6: Cost Comparison — Dev vs CP-6 ───────────────────────────────────
lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{Cost savings comparison between the development corpus (CP-5R) and external validation (CP-6), both at the gate tier with $\alpha = 0.10$.}")
lines.append(r"\label{tab:cost_comparison}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{lrr}")
lines.append(r"\toprule")
lines.append(r"Metric & Dev (n=208) & CP-6 (n=89) \\")
lines.append(r"\midrule")
dev_cost = cp5_eval['cost_alpha_0.10']
cp6_cost = cp6_eval['tier1_savings']
lines.append(f"Gross savings (\\%) & {dev_cost['gross_savings_pct']:.2f} & {cp6_cost['gross_savings_pct']:.2f} \\\\")
lines.append(f"FP cost (\\%) & {dev_cost['fp_cost_pct']:.2f} & {cp6_cost['fp_cost_pct']:.2f} \\\\")
lines.append(f"Net savings (\\%) & {dev_cost['net_savings_pct']:.2f} & {cp6_cost['net_savings_pct']:.2f} \\\\")
lines.append(f"Oracle ceiling (\\%) & {dev_cost['oracle_savings_pct']:.2f} & {cp6_cost['oracle_ceiling_pct']:.2f} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
(tables_dir / "table6_cost_comparison.tex").write_text('\n'.join(lines))
print("✓ Table 6: Cost comparison")

# ── TABLE 7: Injection Effect Sizes ───────────────────────────────────────────
lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{Effect sizes of synthetic fault injections on validation scores. F1 (RCP) faults are naturally occurring and not injected.}")
lines.append(r"\label{tab:injection_effects}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{lrrrrr}")
lines.append(r"\toprule")
lines.append(r"Fault & $n$ & Mean $\Delta$ & Median $\Delta$ & Min $\Delta$ & Max $\Delta$ \\")
lines.append(r"\midrule")
oe = oracle
f2e = oe['f2_score_inflation']
f3e = oe['f3_score_degradation']
lines.append(f"F2 (DL) & {f2e['n_measured']} & {f2e['mean_delta']:+.4f} & {f2e['median_delta']:+.4f} & {f2e['min_delta']:+.4f} & {f2e['max_delta']:+.4f} \\\\")
lines.append(f"F3 (TBF) & {f3e['n_measured']} & {f3e['mean_delta']:+.4f} & {f3e['median_delta']:+.4f} & {f3e['min_delta']:+.4f} & {f3e['max_delta']:+.4f} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
(tables_dir / "table7_injection_effects.tex").write_text('\n'.join(lines))
print("✓ Table 7: Injection effects")

# ── TABLE 8: Gate vs Monitor vs Oracle ────────────────────────────────────────
lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{Tier comparison on CP-6: Gate (Tier~1, structural features, blocks trials), Monitor (Tier~2, all features, flags for review), and Oracle (ground-truth upper bound).}")
lines.append(r"\label{tab:tier_comparison}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{lrrrrrr}")
lines.append(r"\toprule")
lines.append(r"Tier & FSR & Recall & Prec. & F1 & TP & FP \\")
lines.append(r"\midrule")
g = cp6_eval['crc_evaluation']['gate']
m = cp6_eval['crc_evaluation']['monitor']
lines.append(f"Gate & {g['fsr']:.4f} & {g['recall']:.4f} & {g['precision']:.4f} & {g['f1']:.4f} & {g['tp']} & {g['fp']} \\\\")
lines.append(f"Monitor & {m['fsr']:.4f} & {m['recall']:.4f} & {m['precision']:.4f} & {m['f1']:.4f} & {m['tp']} & {m['fp']} \\\\")
lines.append(f"Oracle & 0.0000 & 1.0000 & 1.0000 & 1.0000 & 54 & 0 \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
(tables_dir / "table8_tier_comparison.tex").write_text('\n'.join(lines))
print("✓ Table 8: Tier comparison")

# ── TABLE 9: CP-6 Trace Summary ──────────────────────────────────────────────
lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{CP-6 AIDE run summary. All runs used OpenAI o3 as the backbone LLM with 20 steps per dataset.}")
lines.append(r"\label{tab:cp6_trace}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{lrrrrrr}")
lines.append(r"\toprule")
lines.append(r"Dataset & Best Metric & Wall (s) & Tokens & Cost (\$) & OK Steps & Buggy \\")
lines.append(r"\midrule")
for _, row in cp6_trace.sort_values('dataset_name').iterrows():
    lines.append(f"{row['dataset_name']} & {row['best_metric']:.4f} & {row['wall_clock_sec']:.0f} & {int(row['total_tokens']):,} & {row['api_cost_usd']:.2f} & {int(row['n_successful_steps'])} & {int(row['n_buggy_steps'])} \\\\")
lines.append(r"\midrule")
tot = cp6_trace[['wall_clock_sec', 'total_tokens', 'api_cost_usd']].sum()
lines.append(f"\\textbf{{Total}} & --- & \\textbf{{{tot['wall_clock_sec']:.0f}}} & \\textbf{{{int(tot['total_tokens']):,}}} & \\textbf{{{tot['api_cost_usd']:.2f}}} & --- & --- \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
(tables_dir / "table9_cp6_trace.tex").write_text('\n'.join(lines))
print("✓ Table 9: CP-6 trace summary")

# ── TABLE 10: LODO Coverage Summary ──────────────────────────────────────────
lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{LODO cross-conformal coverage summary. Each fold holds out one dataset and calibrates on the remaining 15. Coverage = fraction of folds where test FSR $\leq \alpha$.}")
lines.append(r"\label{tab:lodo_summary}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{rrr}")
lines.append(r"\toprule")
lines.append(r"$\alpha$ & Covered Folds & Coverage Rate \\")
lines.append(r"\midrule")
for a_str, stats in cp5_eval['lodo_coverage_summary'].items():
    rate = stats['n_covered'] / stats['n_folds']
    lines.append(f"{float(a_str):.2f} & {stats['n_covered']}/{stats['n_folds']} & {rate:.3f} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
(tables_dir / "table10_lodo_summary.tex").write_text('\n'.join(lines))
print("✓ Table 10: LODO summary")

# ── TABLE 11: Natural Fault Rates (CP-3 Summary) ─────────────────────────────
lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{Natural fault rates observed in the AIDE corpus before synthetic injection (208 non-buggy steps across 16 CC-18 datasets).}")
lines.append(r"\label{tab:natural_faults}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{lrr}")
lines.append(r"\toprule")
lines.append(r"Fault Class & Count & Rate \\")
lines.append(r"\midrule")
nf = cp3_sum['natural_faults']
lines.append(f"F1 (Redundant Config.) & {nf['F1']['count']} & {nf['F1']['rate']:.4f} \\\\")
lines.append(f"F2 (Data Leakage) & {nf['F2']['count']} & {nf['F2']['rate']:.4f} \\\\")
lines.append(f"F3 (Training-Budget) & {nf['F3']['count']} & {nf['F3']['rate']:.4f} \\\\")
lines.append(r"\midrule")
lines.append(f"Any fault & {nf['any']['count']} & {nf['any']['rate']:.4f} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
(tables_dir / "table11_natural_faults.tex").write_text('\n'.join(lines))
print("✓ Table 11: Natural fault rates")

print("\n" + "="*60)
print(f"All figures saved to: {OUT}")
print(f"All tables saved to: {tables_dir}")
print(f"Total: 17 figures (PDF + PNG) and 11 tables (LaTeX)")
print("="*60)
