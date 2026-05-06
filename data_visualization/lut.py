import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 1. Generate 15 random data points
np.random.seed(42)
x = np.random.uniform(-20, 20, 15)
y = np.random.uniform(-20, 20, 15)
# Prediction: Peak 75% efficiency, dropping over distance
eff = 0.75 * np.exp(-(x**2 + y**2) / 400) 

def plot_lut():
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Create a dense grid for interpolation
    grid_x, grid_y = np.mgrid[-25:25:100j, -25:25:100j]
    grid_z = griddata((x, y), eff, (grid_x, grid_y), method='cubic')

    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolors='none', alpha=0.8)
    ax.scatter(x, y, eff, color='red', s=50, label='Measured Points')
    
    ax.set_title('3D Efficiency Lookup Table (Interpolated)')
    ax.set_xlabel('X Misalignment (mm)')
    ax.set_ylabel('Y Misalignment (mm)')
    ax.set_zlabel('Efficiency')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

if __name__ == "__main__":
    plot_lut()
