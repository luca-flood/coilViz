"""
01_efficiency_vs_curvature.py
Efficiency vs sagitta height — core finding.
Run from: coil_viz/data_analysis/
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATA = '../data/wpt_results.csv'
OUT  = './'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': '--',
})

df   = pd.read_csv(DATA)
full = df[df['ablation_name'] == 'full_model']

# Aligned origin only
aligned = full[(full.x_cord==0) & (full.y_cord==0) &
               (full.angle==0)].sort_values('conic_curve_mm')
heights    = aligned['conic_curve_mm'].values
efficiency = aligned['efficiency'].values
baseline   = efficiency[0]
gain       = efficiency - baseline

# ── Figure 1: Line plot ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(heights, efficiency, 'o-', color='#3d4a6b', linewidth=2.5,
        markersize=7, markerfacecolor='white', markeredgewidth=2,
        label='Full model (aligned, θ=0°)')
ax.axhline(baseline, color='#aaaaaa', linestyle='--', linewidth=1.2,
           label=f'Control baseline ({baseline:.3f}%)')
ax.fill_between(heights, baseline, efficiency,
                alpha=0.12, color='#3d4a6b', label='Gain from curvature')
ax.set_xlabel('Sagitta Height h (mm)', fontsize=12)
ax.set_ylabel('Power Transfer Efficiency η (%)', fontsize=12)
ax.set_title('Efficiency vs Coil Curvature (Sagitta Height)\n'
             'Tx at origin, Rx aligned (x=0, y=0, θ=0°), d=50mm',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.annotate(f'+{gain[-1]:.3f}pp ({gain[-1]/baseline*100:.1f}% gain)',
            xy=(heights[-1], efficiency[-1]),
            xytext=(heights[-1]-18, efficiency[-1]+0.05),
            fontsize=9, color='#3d4a6b',
            arrowprops=dict(arrowstyle='->', color='#3d4a6b', lw=1.2))
plt.tight_layout()
plt.savefig(f'{OUT}efficiency_vs_curvature.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved: efficiency_vs_curvature.png')

# ── Figure 2: Bar chart of gain ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.Blues(np.linspace(0.3, 0.85, len(heights)))
bars = ax.bar(heights, gain, width=3.5, color=colors,
              edgecolor='white', linewidth=0.5)
for bar, g in zip(bars, gain):
    if g > 0.005:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f'+{g:.3f}', ha='center', va='bottom', fontsize=7.5,
                color='#1e2a45')
ax.set_xlabel('Sagitta Height h (mm)', fontsize=12)
ax.set_ylabel('Efficiency Gain over Control (pp)', fontsize=12)
ax.set_title('Pure Geometric Contribution of Curvature\n'
             'Efficiency gain relative to flat coil (h=0mm)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}curvature_gain_bar.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved: curvature_gain_bar.png')

# ── Figure 3: All ablations vs curvature ─────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ablations = sorted(df['ablation_name'].unique())
palette = plt.cm.tab10(np.linspace(0, 0.7, len(ablations)))
for abl, color in zip(ablations, palette):
    sub = df[(df.ablation_name==abl) & (df.x_cord==0) &
             (df.y_cord==0) & (df.angle==0)].sort_values('conic_curve_mm')
    lw  = 3.0 if abl == 'full_model' else 1.5
    ls  = '-'  if abl == 'full_model' else '--'
    ax.plot(sub.conic_curve_mm, sub.efficiency, ls,
            color=color, linewidth=lw, label=abl)
ax.set_xlabel('Sagitta Height h (mm)', fontsize=12)
ax.set_ylabel('Efficiency η (%)', fontsize=12)
ax.set_title('Efficiency vs Curvature — All Ablation Configurations\n'
             'Aligned origin (x=0, y=0, θ=0°)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(f'{OUT}ablation_comparison.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved: ablation_comparison.png')

# ── Print summary ─────────────────────────────────────────────
print('\nEfficiency vs Curvature (aligned origin, full model):')
print(f'{"h (mm)":<8} {"η (%)":<10} {"Gain (pp)":<12} {"Rel. Gain"}')
print('-'*44)
for h, e, g in zip(heights, efficiency, gain):
    print(f'{h:<8} {e:<10.5f} {g:<12.5f} {g/baseline*100:.2f}%')
