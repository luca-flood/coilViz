import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def plot_lut2():
    # Simulated data points: Angle (0, 45, 90) and Frequency (90kHz - 110kHz)
    angles = [0, 0, 0, 45, 45, 45, 90, 90, 90]
    freqs = [95, 100, 105, 95, 100, 105, 95, 100, 105]
    # Efficiency drops and shifts peak as angle increases
    eff = [0.6, 0.75, 0.65, 0.4, 0.5, 0.55, 0.1, 0.2, 0.15]

    grid_angle, grid_freq = np.mgrid[0:90:100j, 90:110:100j]
    grid_eff = griddata((angles, freqs), eff, (grid_angle, grid_freq), method='cubic')

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(grid_angle, grid_freq, grid_eff, cmap='magma')
    
    ax.set_title('LUT 2: Angular-Frequency Efficiency Map')
    ax.set_xlabel('Angle (Degrees)')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_zlabel('Efficiency')
    fig.colorbar(surf, label='Efficiency')
    plt.show()

if __name__ == "__main__":
    plot_lut2()

