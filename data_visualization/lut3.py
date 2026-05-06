import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def plot_lut3():
    # Simulated data: How Capacitance needs to change as you move X to stay efficient
    x_off = [0, 0, 0, 10, 10, 10, 20, 20, 20]
    caps = [480, 500, 520, 480, 500, 520, 480, 500, 520]
    # At 20mm, the peak efficiency might require a lower capacitance
    eff = [0.5, 0.8, 0.5, 0.4, 0.6, 0.4, 0.3, 0.2, 0.4] 

    grid_x, grid_cap = np.mgrid[0:20:100j, 480:520:100j]
    grid_eff = griddata((x_off, caps), eff, (grid_x, grid_cap), method='cubic')

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(grid_x, grid_cap, grid_eff, cmap='viridis')
    
    ax.set_title('LUT 3: Capacitance Compensation Map')
    ax.set_xlabel('X Offset (mm)')
    ax.set_ylabel('Variable Capacitance (pF)')
    ax.set_zlabel('Efficiency')
    fig.colorbar(surf)
    plt.show()

if __name__ == "__main__":
    plot_lut3()

