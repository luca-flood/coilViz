"""
06_coupling_analysis.py
Coupling coefficient, kQ regime, frequency splitting.
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

df      = pd.read_csv(DATA)
full    = df[df['ablation_name'] == 'full_model']
aligned = full[(full.x_cord==0) & (full.y_cord==0) &
               (full.angle==0)].sort_values('conic_curve_mm')

# Figure 1: k and kQ vs curvature
fig, ax1 = plt.subplots(figsize=(9, 5.5))
ax2 = ax1.twinx()
l1, = ax1.plot(aligned.conic_curve_mm, aligned.k, 'o-',
               color='#3d4a6b', linewidth=2.5, markersize=7,
               markerfacecolor='white', markeredgewidth=2, label='k')
l2, = ax2.plot(aligned.conic_curve_mm, aligned.kQ, 's--',
               color='#e05c4b', linewidth=2, markersize=6,
               markerfacecolor='white', markeredgewidth=2, label='kQ')
ax2.axhline(1.0, color='#e05c4b', linestyle=':', alpha=0.5,
            label='kQ=1 (critical)')
ax1.set_xlabel('Sagitta Height h (mm)', fontsize=12)
ax1.set_ylabel('Coupling Coefficient k', fontsize=12, color='#3d4a6b')
ax2.set_ylabel('kQ', fontsize=12, color='#e05c4b')
ax1.tick_params(axis='y', labelcolor='#3d4a6b')
ax2.tick_params(axis='y', labelcolor='#e05c4b')
ax1.legend([l1,l2], ['k','kQ'], fontsize=10)
ax1.set_title('Coupling Coefficient and kQ vs Sagitta Height\nAligned origin, d=50mm',
              fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}k_vs_curvature.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved: k_vs_curvature.png')

# Figure 2: System operating point on efficiency curve
fig, ax = plt.subplots(figsize=(9, 5.5))
kQ_r  = np.linspace(0.01, aligned.kQ.max()*1.3, 500)
eta_c = (kQ_r**2/(1+kQ_r**2)**2)*100
ax.plot(kQ_r, eta_c, '-', color='#888', linewidth=2, alpha=0.7,
        label='η = (kQ)²/(1+(kQ)²)²')
ax.axvline(1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.8,
           label='kQ=1: η_max=25%')
ax.axhline(25, color='green', linestyle=':', alpha=0.4)
sc = ax.scatter(aligned.kQ, aligned.efficiency,
                c=aligned.conic_curve_mm, cmap='Blues',
                s=80, zorder=5, edgecolors='#3d4a6b', linewidths=0.8,
                label='Your system')
plt.colorbar(sc, ax=ax, label='Sagitta h (mm)')
ax.set_xlabel('kQ', fontsize=12); ax.set_ylabel('Efficiency η (%)', fontsize=12)
ax.set_title('System Operating Point on Efficiency Curve\nEach point = one sagitta height',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.annotate('Overcoupled\nregion', xy=(aligned.kQ.mean(), 1.0),
            fontsize=9, color='#555', ha='center')
plt.tight_layout()
plt.savefig(f'{OUT}kQ_regime_plot.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved: kQ_regime_plot.png')

# Figure 3: Frequency splitting
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(aligned.conic_curve_mm, aligned.split_gap_kHz, 'o-',
        color='#3d4a6b', linewidth=2.5, markersize=7,
        markerfacecolor='white', markeredgewidth=2)
ax.fill_between(aligned.conic_curve_mm, 0, aligned.split_gap_kHz,
                alpha=0.1, color='#3d4a6b')
ax.set_xlabel('Sagitta Height h (mm)', fontsize=12)
ax.set_ylabel('Frequency Split Gap (kHz)', fontsize=12)
ax.set_title('Frequency Splitting vs Sagitta Height\nAligned origin — gap = f₊ − f₋',
             fontsize=13, fontweight='bold')
mx = aligned.split_gap_kHz.max()
mh = aligned.loc[aligned.split_gap_kHz.idxmax(), 'conic_curve_mm']
ax.annotate(f'{mx:.1f} kHz', xy=(mh, mx),
            xytext=(mh-12, mx+15), fontsize=9, color='#3d4a6b',
            arrowprops=dict(arrowstyle='->', color='#3d4a6b'))
plt.tight_layout()
plt.savefig(f'{OUT}frequency_splitting.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved: frequency_splitting.png')

# Print coupling summary
print('\nCoupling summary (aligned origin):')
print(f'{"h":>6} {"k":>10} {"kQ":>10} {"η%":>10} {"regime":>14} {"split kHz":>12}')
print('-'*64)
for _, row in aligned.iterrows():
    print(f'{row.conic_curve_mm:>6.0f} {row.k:>10.6f} {row.kQ:>10.4f} '
          f'{row.efficiency:>10.4f} {row.coupling_regime:>14} '
          f'{row.split_gap_kHz:>12.1f}')
