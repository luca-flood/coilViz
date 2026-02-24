import numpy as np
import matplotlib.pyplot as plt

def generate_mock_data(conic_mm, peak_eff, decay_rate):
    """Generates 15 random X,Y points with predicted efficiency for a specific coil."""
    np.random.seed(conic_mm) # Keep points distinct but reproducible
    x = np.random.uniform(-20, 20, 15)
    y = np.random.uniform(-20, 20, 15)
    
    # Radial distance R = sqrt(x^2 + y^2)
    r = np.sqrt(x**2 + y**2)
    
    # Efficiency model: Higher conic height = better reach (slower decay)
    eff = peak_eff * np.exp(-(r**2) / decay_rate)
    return r, eff

def plot_multi_curve_graph():
    plt.figure(figsize=(10, 6))
    
    # Define the different conic heights you designed in Fusion 360
    # Format: (Height_mm, Peak_Efficiency, Decay_Factor)
    coil_configs = [
        (10, 0.50, 200),  # Shallow curve: lower peak, fast drop
        (25, 0.65, 400),  # Medium curve
        (45, 0.75, 700)   # Deep curve: higher peak, stays efficient longer
    ]
    
    colors = ['#e74c3c', '#3498db', '#2ecc71'] # Red, Blue, Green
    
    for i, (mm, peak, decay) in enumerate(coil_configs):
        r, eff = generate_mock_data(mm, peak, decay)
        
        # Sort values by distance so the line plot doesn't zig-zag
        sort_idx = np.argsort(r)
        
        plt.plot(r[sort_idx], eff[sort_idx], 
                 label=f'Conic: {mm}mm', 
                 color=colors[i], 
                 marker='o', 
                 markersize=4, 
                 linewidth=2, 
                 alpha=0.8)

    # Formatting the chart
    plt.title('Efficiency vs. Displacement for Multiple Conic Geometries', fontsize=14)
    plt.xlabel('Total Radial Misalignment (mm)', fontsize=12)
    plt.ylabel('Efficiency (Decimal %)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Coil Curvature", frameon=True)
    
    # Set Y-axis to show 0 to 100% scale clearly
    plt.ylim(0, 1.0) 
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_multi_curve_graph()
