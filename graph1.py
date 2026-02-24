import numpy as np
import matplotlib.pyplot as plt

def plot_graph1():
    plt.figure(figsize=(9, 6))
    # Simulated frequencies around a 100kHz resonance
    freqs = np.linspace(80, 120, 50) 
    
    # Different curves show how the "Q-factor" changes with misalignment
    configs = [("Perfect Alignment", 0.85, 5), ("5mm Misaligned", 0.65, 8), ("10mm Misaligned", 0.45, 12)]
    
    for label, peak, width in configs:
        # Lorentizian-style resonance curve
        eff = peak / (1 + ((freqs - 100) / width)**2)
        plt.plot(freqs, eff, label=label, linewidth=2)

    plt.title("Graph 1: Frequency vs. Efficiency (Resonant Drift)")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Efficiency (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    plot_graph1()
