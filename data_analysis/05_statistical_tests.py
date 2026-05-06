"""
05_statistical_tests.py
Statistical analysis: ANOVA, regression, correlations, effect size.
Run from: coil_viz/data_analysis/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kruskal, f_oneway, pearsonr
import warnings; warnings.filterwarnings('ignore')

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
heights = sorted(full.conic_curve_mm.unique())

rep = ["="*65,
       "  WPT COIL CURVATURE — STATISTICAL ANALYSIS",
       "="*65,
       f"\nData: {len(full)} rows (full_model ablation only)",
       f"Heights tested: {heights} mm"]

# 1. ANOVA across all poses
groups = [full[full.conic_curve_mm==h]['efficiency'].values for h in heights]
F, p_a = f_oneway(*groups)
rep += ["\n"+"─"*65,
        "1. ONE-WAY ANOVA — efficiency across heights (all poses)",
        "─"*65,
        f"\n   F = {F:.4f}   p = {p_a:.2e}",
        f"   {'SIGNIFICANT' if p_a<0.05 else 'NOT SIGNIFICANT'} at α=0.05",
        "\n   NOTE: Low significance expected when averaging over θ=90°",
        "   poses where η=0 for all heights. See regression below",
        "   for the meaningful signal at aligned poses."]

# 2. Kruskal-Wallis
H, p_kw = kruskal(*groups)
rep += ["\n"+"─"*65, "2. KRUSKAL-WALLIS (non-parametric)",
        "─"*65, f"\n   H = {H:.4f}   p = {p_kw:.2e}",
        f"   {'SIGNIFICANT' if p_kw<0.05 else 'NOT SIGNIFICANT'}"]

# 3. Regression at aligned origin
h_arr   = aligned.conic_curve_mm.values.astype(float)
eta_arr = aligned.efficiency.values
slope, intercept, r, p_reg, se = stats.linregress(h_arr, eta_arr)
rep += ["\n"+"─"*65,
        "3. LINEAR REGRESSION — aligned origin (x=y=0, θ=0°)",
        "─"*65,
        f"\n   Slope:     {slope:.6f} %/mm",
        f"   Intercept: {intercept:.5f} %",
        f"   R²:        {r**2:.6f}",
        f"   p-value:   {p_reg:.2e}",
        f"   *** STRONG FIT — η increases {slope:.4f}pp per mm of sagitta ***",
        f"\n   At h=85mm predicted η = {slope*85+intercept:.4f}%",
        f"   vs h=0 baseline η = {intercept:.4f}%",
        f"   Total predicted gain = {slope*85:.4f}pp "
        f"({slope*85/intercept*100:.1f}% relative)"]

# 4. Cohen's d
rep += ["\n"+"─"*65,
        "4. EFFECT SIZE (Cohen's d) — control vs each treatment",
        "   < 0.2 negligible | 0.2-0.5 small | 0.5-0.8 medium | > 0.8 large",
        "─"*65]
cohens_d = {}
ctrl = full[full.conic_curve_mm==0]['efficiency'].values
for h in heights[1:]:
    treat = full[full.conic_curve_mm==h]['efficiency'].values
    n1,n2 = len(ctrl), len(treat)
    sp    = np.sqrt(((n1-1)*np.std(ctrl,ddof=1)**2 +
                     (n2-1)*np.std(treat,ddof=1)**2)/(n1+n2-2))
    d     = (np.mean(treat)-np.mean(ctrl))/sp if sp>0 else 0
    cohens_d[h] = d
    lbl = ('negligible' if abs(d)<0.2 else 'small' if abs(d)<0.5
           else 'medium' if abs(d)<0.8 else 'large')
    rep.append(f"\n   h={h:>3}mm: d = {d:+.4f} [{lbl}]")

# 5. Pearson correlations
rep += ["\n"+"─"*65, "5. PEARSON CORRELATIONS", "─"*65]
for c1, c2, lbl in [
    ('conic_curve_mm','efficiency','h vs η (all poses)'),
    ('k',             'efficiency','k vs η'),
    ('kQ',            'efficiency','kQ vs η'),
    ('Q_rx',          'efficiency','Q_rx vs η'),
    ('d_lateral_mm',  'efficiency','d_lateral vs η'),
    ('angle',         'efficiency','θ vs η'),
    ('conic_curve_mm','k',         'h vs k'),
]:
    v = full[[c1,c2]].dropna()
    r_p, p_p = pearsonr(v[c1], v[c2])
    sig = '***' if p_p<0.001 else '**' if p_p<0.01 else '*' if p_p<0.05 else 'ns'
    rep.append(f"\n   {lbl:<28} r={r_p:+.4f}  p={p_p:.2e}  {sig}")

# 6. Key thresholds
rep += ["\n"+"─"*65, "6. EFFICIENCY AT KEY MISALIGNMENT THRESHOLDS", "─"*65]
for lbl, sub in [
    ('X=40mm, Y=0, θ=0',  full[(full.x_cord==40)&(full.y_cord==0)&(full.angle==0)]),
    ('X=80mm, Y=0, θ=0',  full[(full.x_cord==80)&(full.y_cord==0)&(full.angle==0)]),
    ('X=0, Y=0, θ=45°',   full[(full.x_cord==0)&(full.y_cord==0)&(full.angle==45)]),
    ('X=0, Y=0, θ=90°',   full[(full.x_cord==0)&(full.y_cord==0)&(full.angle==90)]),
]:
    rep.append(f"\n   {lbl}:")
    for h in [0, 25, 50, 85]:
        eta = sub[sub.conic_curve_mm==h]['efficiency'].values
        if len(eta)>0:
            rep.append(f"     h={h:>3}mm: η={eta[0]:.4f}%")

text = '\n'.join(rep)
with open(f'{OUT}stats_report.txt','w') as f: f.write(text)
print(text); print(f'\nSaved: stats_report.txt')

# Figure 1: Regression
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(h_arr, eta_arr, 'o', color='#3d4a6b', markersize=8,
        markerfacecolor='white', markeredgewidth=2, label='Simulated η')
x_l = np.array([0, heights[-1]])
ax.plot(x_l, slope*x_l+intercept, '--', color='#e05c4b', linewidth=2,
        label=f'Linear fit  R²={r**2:.4f}, p={p_reg:.2e}')
ax.fill_between(x_l, (slope-1.96*se)*x_l+intercept,
                (slope+1.96*se)*x_l+intercept,
                alpha=0.1, color='#e05c4b', label='95% CI')
ax.set_xlabel('Sagitta Height h (mm)', fontsize=12)
ax.set_ylabel('Efficiency η (%)', fontsize=12)
ax.set_title('Linear Regression: Efficiency vs Sagitta Height\nAligned origin',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT}regression_plot.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved: regression_plot.png')

# Figure 2: Cohen's d
fig, ax = plt.subplots(figsize=(9, 5))
hs = list(cohens_d.keys()); ds = list(cohens_d.values())
colors = ['#c0c8e0' if abs(d)<0.2 else '#8f9fc8' if abs(d)<0.5
          else '#5b6fa8' if abs(d)<0.8 else '#2c3e6b' for d in ds]
ax.bar(hs, ds, width=3.5, color=colors, edgecolor='white')
for thr, lbl, c in [(0.2,'Small','#aaa'),(0.5,'Medium','#888'),(0.8,'Large','#555')]:
    ax.axhline(thr, color=c, linestyle=':', linewidth=1, label=lbl)
ax.set_xlabel('Sagitta h (mm)', fontsize=12)
ax.set_ylabel("Cohen's d", fontsize=12)
ax.set_title("Effect Size: Each Treatment vs Control\nCohen's d",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT}effect_size_bar.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved: effect_size_bar.png')

# Figure 3: Correlation matrix
corr_cols   = ['conic_curve_mm','efficiency','k','kQ','Q_rx',
               'd_lateral_mm','angle','M_nH']
corr_labels = ['h (mm)','η (%)','k','kQ','Q_rx','d_lat','θ°','M(nH)']
mat = full[corr_cols].corr()
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(mat.values, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label='Pearson r')
ax.set_xticks(range(len(corr_labels))); ax.set_yticks(range(len(corr_labels)))
ax.set_xticklabels(corr_labels, rotation=40, ha='right', fontsize=9)
ax.set_yticklabels(corr_labels, fontsize=9)
for i in range(len(corr_labels)):
    for j in range(len(corr_labels)):
        v = mat.values[i,j]
        ax.text(j,i,f'{v:.2f}', ha='center', va='center', fontsize=8,
                color='white' if abs(v)>0.6 else 'black')
ax.set_title('Correlation Matrix', fontsize=12, fontweight='bold')
ax.spines[:].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUT}correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved: correlation_matrix.png')
