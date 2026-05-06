import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def plot_heatmap1():
    # 15 random XY points
    np.random.seed(99)
    x = np.random.uniform(-20, 20, 15)
    y = np.random.uniform(-20, 20, 15)
    # Temp prediction: Higher temp when misaligned (higher current/losses)
    temp = 25 + 15 * (np.sqrt(x**2 + y**2) / 20) 

    grid_x, grid_y = np.mgrid[-25:25:200j, -25:25:200j]
    grid_z = griddata((x, y), temp, (grid_x, grid_y), method='cubic')

    plt.figure(figsize=(8, 7))
    plt.contourf(grid_x, grid_y, grid_z, levels=50, cmap='inferno')
    plt.colorbar(label='Coil Temperature (°C)')
    plt.title("Heatmap 1: Thermal Profile of Misalignment")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.show()

if __name__ == "__main__":
    plot_heatmap1()
