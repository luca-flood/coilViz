"""
02_efficiency_heatmap.py
2D efficiency heatmap over X/Y pose space.
Run from: coil_viz/data_analysis/
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATA = '../data/wpt_results.csv'
OUT  = './'

plt.rcParams.update({'font.family': 'DejaVu Sans'})

df   = pd.read_csv(DATA)
full = df[df['ablation_name'] == 'full_model']

def make_grid(df_sub, angle=0):
    sub    = df_sub[df_sub.angle == angle]
    x_vals = sorted(sub.x_cord.unique())
    y_vals = sorted(sub.y_cord.unique())
    grid   = np.full((len(y_vals), len(x_vals)), np.nan)
    for _, row in sub.iterrows():
        if row.x_cord in x_vals and row.y_cord in y_vals:
            xi = x_vals.index(row.x_cord)
            yi = y_vals.index(row.y_cord)
            grid[yi, xi] = row.efficiency
    return np.array(x_vals), np.array(y_vals), grid

heights_to_plot = [0, 25, 50, 85]

# Compute shared vmax
vmax = max(
    np.nanmax(make_grid(full[full.conic_curve_mm==h])[2])
    for h in heights_to_plot
)

# Individual heatmaps
for h in heights_to_plot:
    sub = full[full.conic_curve_mm == h]
    x_vals, y_vals, grid = make_grid(sub)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, origin='lower', aspect='auto',
                   extent=[x_vals[0]-2.5, x_vals[-1]+2.5,
                           y_vals[0]-2.5, y_vals[-1]+2.5],
                   cmap='YlOrRd', interpolation='nearest',
                   vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Efficiency η (%)')
    ax.set_xlabel('X Lateral Offset (mm)', fontsize=11)
    ax.set_ylabel('Y Lateral Offset (mm)', fontsize=11)
    grp = 'Control' if h == 0 else 'Treatment'
    ax.set_title(f'Efficiency Heatmap — h={h}mm ({grp})\nθ=0°, d=50mm',
                 fontsize=12, fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{OUT}heatmap_h{h}mm.png', dpi=150, bbox_inches='tight')
    plt.close(); print(f'Saved: heatmap_h{h}mm.png')

# Side-by-side: flat vs max curvature
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, h in zip(axes, [0, 85]):
    sub = full[full.conic_curve_mm == h]
    x_vals, y_vals, grid = make_grid(sub)
    im = ax.imshow(grid, origin='lower', aspect='auto',
                   extent=[x_vals[0]-2.5, x_vals[-1]+2.5,
                           y_vals[0]-2.5, y_vals[-1]+2.5],
                   cmap='YlOrRd', interpolation='nearest',
                   vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label='η (%)')
    label = 'Control (flat, h=0mm)' if h == 0 else f'Max curvature (h={h}mm)'
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_xlabel('X Offset (mm)', fontsize=10)
    ax.set_ylabel('Y Offset (mm)', fontsize=10)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Efficiency Heatmap — Flat vs Maximum Curvature\n'
             'θ=0°, d=50mm, shared color scale', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}heatmap_comparison.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved: heatmap_comparison.png')
