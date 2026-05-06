import numpy as np
import matplotlib.pyplot as plt

def plot_graph2():
    # Conic curve heights from your Fusion 360 models
    conic_heights = np.array([10, 15, 20, 25, 30, 35, 40, 45])
    
    # As the curve gets deeper, Inductance (L) usually changes
    # C = 1 / ( (2*pi*f)^2 * L )
    # We simulate the required pF to stay at 100kHz
    required_cap = 500 - (conic_heights * 5) 

    plt.figure(figsize=(8, 5))
    plt.step(conic_heights, required_cap, where='mid', color='purple', linewidth=2)
    plt.scatter(conic_heights, required_cap, color='black')
    
    plt.title("Graph 2: Required Capacitance vs. Conic Height")
    plt.xlabel("Conic Curve Height (mm)")
    plt.ylabel("Capacitance (pF) for 100kHz Resonance")
    plt.grid(axis='y', linestyle='--')
    plt.show()

if __name__ == "__main__":
    plot_graph2()
