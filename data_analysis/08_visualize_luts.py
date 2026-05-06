"""
08_visualize_luts.py
─────────────────────
Interactive 3D surface plots of the LUTs.
Two windows:
  1. η vs X-offset × Sagitta height
  2. η vs Angle × Sagitta height

Controls:
  - Click and drag to rotate
  - Scroll to zoom
  - Right-click drag to pan
  - Close window to exit

Run from: coil_viz/data_analysis/
Requires: lut_pivot_xh.csv, lut_pivot_angleh.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import RectBivariateSpline

OUT = './'

# ── Load LUTs ─────────────────────────────────────────────────
lut_xh  = pd.read_csv(f'{OUT}lut_pivot_xh.csv',    index_col=0)
lut_ah  = pd.read_csv(f'{OUT}lut_pivot_angleh.csv', index_col=0)

x_vals  = lut_xh.index.values.astype(float)
h_vals  = lut_xh.columns.values.astype(float)
a_vals  = lut_ah.index.values.astype(float)
h_vals2 = lut_ah.columns.values.astype(float)

Z_xh = lut_xh.values.astype(float)   # shape: (n_x, n_h)
Z_ah = lut_ah.values.astype(float)   # shape: (n_a, n_h)


def make_surface(x_raw, y_raw, Z_raw, x_res=200, y_res=200):
    """Bicubic interpolation onto a fine grid for smooth surface."""
    spline = RectBivariateSpline(x_raw, y_raw, Z_raw, kx=3, ky=3)
    xi = np.linspace(x_raw[0], x_raw[-1], x_res)
    yi = np.linspace(y_raw[0], y_raw[-1], y_res)
    Zi = spline(xi, yi)
    Zi = np.clip(Zi, 0, None)   # no negative efficiency
    return xi, yi, Zi


def plot_surface(x_raw, y_raw, Z_raw,
                 xlabel, ylabel, title,
                 x_res=200, y_res=200,
                 cmap='plasma'):
    xi, yi, Zi = make_surface(x_raw, y_raw, Z_raw, x_res, y_res)
    XX, YY = np.meshgrid(xi, yi, indexing='ij')

    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(XX, YY, Zi,
                           cmap=cmap,
                           linewidth=0,
                           antialiased=True,
                           alpha=0.92,
                           rcount=200, ccount=200)

    # Contour projected onto floor
    offset = Zi.min() - Zi.max() - Zi.min() * 0.08
    ax.contourf(XX, YY, Zi, zdir='z', offset=offset,
                levels=20, cmap=cmap, alpha=0.35)

    cb = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.08)
    cb.set_label('Efficiency η (%)', fontsize=11)

    ax.set_xlabel(xlabel, fontsize=11, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=10)
    ax.set_zlabel('Efficiency η (%)', fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

    # Mark the peak
    peak_idx = np.unravel_index(np.argmax(Zi), Zi.shape)
    px, py, pz = XX[peak_idx], YY[peak_idx], Zi[peak_idx]
    ax.scatter([px], [py], [pz], color='white', s=80,
               edgecolors='black', linewidths=1.5, zorder=10)
    ax.text(px, py, pz + Zi.max() - Zi.min()*0.04,
            f'peak\n{pz:.3f}%',
            fontsize=8, ha='center', color='white',
            fontweight='bold')

    ax.view_init(elev=30, azim=-60)

    # Instructions in corner
    fig.text(0.01, 0.01,
             'Drag to rotate  |  Scroll to zoom  |  Right-drag to pan',
             fontsize=8, color='#666666')

    return fig, ax


# ── Surface 1: X offset × Sagitta height ─────────────────────
print('Building surface 1: X offset × sagitta height...')
fig1, ax1 = plot_surface(
    x_vals, h_vals, Z_xh,
    xlabel='X Lateral Offset (mm)',
    ylabel='Sagitta Height h (mm)',
    title='Efficiency Surface — X Offset × Sagitta Height\n'
          'Y=0, θ=0°, d=50mm axial separation',
    cmap='plasma'
)

# ── Surface 2: Angle × Sagitta height ────────────────────────
print('Building surface 2: angle × sagitta height...')
fig2, ax2 = plot_surface(
    a_vals, h_vals2, Z_ah,
    xlabel='Angular Misalignment θ (°)',
    ylabel='Sagitta Height h (mm)',
    title='Efficiency Surface — Angular Misalignment × Sagitta Height\n'
          'x=0, y=0, d=50mm axial separation',
    cmap='viridis'
)

# ── Surface 3: X offset × Angle at h=0 and h=85 side by side ─
print('Building surface 3: X offset × Angle comparison...')

# Build X vs Angle grid from full LUT
lut_full = pd.read_csv(f'{OUT}lut_full.csv')
lut_full = lut_full[lut_full['alignment'] == 'x-axis']

for h_show, cmap_show, label in [(0, 'Blues', 'Control (h=0mm)'),
                                   (85, 'Reds', 'Max Curvature (h=85mm)')]:
    sub = lut_full[lut_full['sagitta_h_mm'] == h_show]
    if len(sub) == 0:
        continue
    pivot = sub.pivot_table(
        values='efficiency_pct', index='x_mm', columns='angle_deg'
    ).fillna(0)
    xv = pivot.index.values.astype(float)
    av = pivot.columns.values.astype(float)
    Zv = pivot.values.astype(float)

    if len(xv) < 4 or len(av) < 4:
        continue

    fig, ax = plot_surface(
        xv, av, Zv,
        xlabel='X Lateral Offset (mm)',
        ylabel='Angle θ (°)',
        title=f'Efficiency Surface — {label}\n'
               'X Offset × Angular Misalignment, d=50mm',
        cmap=cmap_show
    )

print('\nAll surfaces ready.')
print('Interact: drag=rotate, scroll=zoom, right-drag=pan')
print('Close any window to move to the next.')
plt.show()
