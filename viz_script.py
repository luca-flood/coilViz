import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata

# 1. SETUP DATA (Mimicking your screenshot)
# Replace this section with: df = pd.read_csv('your_data.csv')
data = {
    'conic_mm': [10, 15, 20, 25, 30, 35, 40, 45] * 3,
    'misalignment_mm': [0]*8 + [5]*8 + [10]*8,
    'efficiency': [
        0.40, 0.43, 0.54, 0.55, 0.61, 0.57, 0.65, 0.72, # 0mm
        0.38, 0.41, 0.50, 0.52, 0.58, 0.54, 0.62, 0.69, # 5mm 
        0.30, 0.34, 0.42, 0.44, 0.50, 0.46, 0.55, 0.60  # 10mm
    ]
}
df = pd.DataFrame(data)

def plot_coil_data(df):
    fig = plt.figure(figsize=(18, 12))
    
    # --- VISUALIZATION 1: LINE GRAPH (Efficiency vs Misalignment) ---
    ax1 = fig.add_subplot(2, 2, 1)
    for mm in df['conic_mm'].unique():
        subset = df[df['conic_mm'] == mm]
        ax1.plot(subset['misalignment_mm'], subset['efficiency'], marker='o', label=f'{mm}mm Curve')
    
    ax1.set_title("Efficiency vs. Misalignment (By Curvature)")
    ax1.set_xlabel("Misalignment (mm)")
    ax1.set_ylabel("Efficiency (%)")
    ax1.legend(title="Conic Height")
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- VISUALIZATION 2: 3D INTERPOLATED SURFACE (The LUT) ---
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    
    # Create grid for interpolation
    xi = np.linspace(df['misalignment_mm'].min(), df['misalignment_mm'].max(), 50)
    yi = np.linspace(df['conic_mm'].min(), df['conic_mm'].max(), 50)
    X, Y = np.meshgrid(xi, yi)
    
    # Interpolate Z (Efficiency) values
    Z = griddata((df['misalignment_mm'], df['conic_mm']), df['efficiency'], (X, Y), method='cubic')
    
    surf = ax2.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.8)
    ax2.set_xlabel('Misalignment (mm)')
    ax2.set_ylabel('Conic Curve (mm)')
    ax2.set_zlabel('Efficiency')
    ax2.set_title('3D Efficiency Lookup Surface')
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)

    # --- VISUALIZATION 3: THE "DONUT" HEATMAP (XY Plane) ---
    # We'll simulate the XY plane for a specific curvature (e.g., 45mm)
    ax3 = fig.add_subplot(2, 2, 3)
    best_curve = df['conic_mm'].max()
    curve_data = df[df['conic_mm'] == best_curve]
    
    # Create a 2D grid representing the space around the Tx coil
    grid_size = np.linspace(-15, 15, 100)
    XX, YY = np.meshgrid(grid_size, grid_size)
    RR = np.sqrt(XX**2 + YY**2) # Radial distance from center
    
    # Interpolate efficiency based on radial distance
    # (Assuming X and Y misalignment behavior is similar)
    EE = np.interp(RR, curve_data['misalignment_mm'], curve_data['efficiency'])
    
    heatmap = ax3.contourf(XX, YY, EE, levels=20, cmap='RdYlBu_r')
    ax3.set_title(f"Efficiency Heatmap (Donut) - {best_curve}mm Curve")
    ax3.set_aspect('equal')
    plt.colorbar(heatmap, ax=ax3, label='Efficiency')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_coil_data(df)
