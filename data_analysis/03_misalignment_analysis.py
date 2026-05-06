"""
03_misalignment_analysis.py
Efficiency vs lateral/angular/diagonal misalignment.
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

HEIGHTS = [0, 25, 50, 85]
PALETTE = ['#1e2a45', '#4a6fa5', '#8fafd4', '#c5d5e8']
LABELS  = {0: 'h=0mm (control)', 25: 'h=25mm',
           50: 'h=50mm', 85: 'h=85mm (max)'}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Efficiency Under Misalignment — Control vs Treatment\nd=50mm axial separation',
             fontsize=14, fontweight='bold')

# X-axis lateral
ax = axes[0, 0]
for h, c in zip(HEIGHTS, PALETTE):
    sub = full[(full.conic_curve_mm==h) & (full.alignment=='x-axis') &
               (full.angle==0)].sort_values('x_cord')
    ax.plot(sub.x_cord, sub.efficiency, 'o-', color=c,
            linewidth=2, markersize=4, label=LABELS[h])
ax.set_xlabel('X Offset (mm)'); ax.set_ylabel('η (%)')
ax.set_title('Lateral X Misalignment', fontweight='bold')
ax.legend(fontsize=8)

# Y-axis lateral
ax = axes[0, 1]
for h, c in zip(HEIGHTS, PALETTE):
    sub = full[(full.conic_curve_mm==h) & (full.alignment=='y-axis') &
               (full.angle==0)].sort_values('y_cord')
    ax.plot(sub.y_cord, sub.efficiency, 'o-', color=c,
            linewidth=2, markersize=4, label=LABELS[h])
ax.set_xlabel('Y Offset (mm)'); ax.set_ylabel('η (%)')
ax.set_title('Lateral Y Misalignment', fontweight='bold')
ax.legend(fontsize=8)

# Angular
ax = axes[1, 0]
for h, c in zip(HEIGHTS, PALETTE):
    sub = full[(full.conic_curve_mm==h) & (full.x_cord==0) &
               (full.y_cord==0)].sort_values('angle')
    ax.plot(sub.angle, sub.efficiency, 'o-', color=c,
            linewidth=2, markersize=4, label=LABELS[h])
ax.set_xlabel('Angle θ (°)'); ax.set_ylabel('η (%)')
ax.set_title('Angular Misalignment', fontweight='bold')
ax.legend(fontsize=8); ax.set_xticks([0, 30, 60, 90])

# Diagonal
ax = axes[1, 1]
for h, c in zip(HEIGHTS, PALETTE):
    sub = full[(full.conic_curve_mm==h) & (full.alignment=='diagonal') &
               (full.angle==0)].sort_values('d_lateral_mm')
    ax.plot(sub.d_lateral_mm, sub.efficiency, 'o-', color=c,
            linewidth=2, markersize=4, label=LABELS[h])
ax.set_xlabel('Diagonal Offset (mm)'); ax.set_ylabel('η (%)')
ax.set_title('Diagonal Misalignment (X=Y)', fontweight='bold')
ax.legend(fontsize=8)

for a in axes.flat:
    a.spines[['top','right']].set_visible(False)
    a.grid(True, alpha=0.25, linestyle='--')

plt.tight_layout()
plt.savefig(f'{OUT}misalignment_combined.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved: misalignment_combined.png')

# Individual high-res plots
for fname, title, xlab, getter in [
    ('lateral_misalignment.png', 'Lateral X Misalignment\nY=0, θ=0°, d=50mm',
     'X Offset (mm)',
     lambda h: full[(full.conic_curve_mm==h)&(full.alignment=='x-axis')&
                    (full.angle==0)].sort_values('x_cord')),
    ('angular_misalignment.png', 'Angular Misalignment\nx=0, y=0, d=50mm',
     'Angle θ (°)',
     lambda h: full[(full.conic_curve_mm==h)&(full.x_cord==0)&
                    (full.y_cord==0)].sort_values('angle')),
    ('diagonal_misalignment.png', 'Diagonal Misalignment (X=Y)\nθ=0°, d=50mm',
     'Diagonal Offset (mm)',
     lambda h: full[(full.conic_curve_mm==h)&(full.alignment=='diagonal')&
                    (full.angle==0)].sort_values('d_lateral_mm')),
]:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for h, c in zip(HEIGHTS, PALETTE):
        sub = getter(h)
        xcol = sub.columns[sub.columns.isin(
            ['x_cord','y_cord','angle','d_lateral_mm'])][0]
        ax.plot(sub[xcol], sub.efficiency, 'o-', color=c,
                linewidth=2.2, markersize=5, label=LABELS[h])
    ax.set_xlabel(xlab, fontsize=12); ax.set_ylabel('Efficiency η (%)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{OUT}{fname}', dpi=150, bbox_inches='tight')
    plt.close(); print(f'Saved: {fname}')
