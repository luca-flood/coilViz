import numpy as np
import matplotlib.pyplot as plt

def plot_graph3():
    # Simulated temp readings during a long test run
    temp = np.linspace(25, 65, 20)
    # Current might drop as resistance increases with temp
    current = 2.0 * (1 - 0.004 * (temp - 25)) 

    plt.figure(figsize=(8, 5))
    plt.plot(temp, current, 'r-o', label='Measured Current')
    plt.title("Graph 3: Current Drop-off due to Thermal Heating")
    plt.xlabel("Ambient/Coil Temperature (°F)")
    plt.ylabel("Output Current (A)")
    plt.axvspan(55, 65, color='red', alpha=0.1, label='Overheat Zone')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_graph3()

