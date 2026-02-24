import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Generate same data points
np.random.seed(42)
x = np.random.uniform(-20, 20, 15)
y = np.random.uniform(-20, 20, 15)
eff = 0.75 * np.exp(-(x**2 + y**2) / 400)

def plot_heatmap():
    plt.figure(figsize=(8, 7))
    
    # Create grid
    grid_x, grid_y = np.mgrid[-25:25:200j, -25:25:200j]
    grid_z = griddata((x, y), eff, (grid_x, grid_y), method='cubic')

    # Plot contour heatmap
    content = plt.contourf(grid_x, grid_y, grid_z, levels=50, cmap='RdYlBu_r')
    plt.scatter(x, y, c='black', s=10, alpha=0.5) # Show where data was taken
    
    plt.title('Spatial Efficiency Heatmap (Top-Down)')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.colorbar(content, label='Efficiency')
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    plot_heatmap()
