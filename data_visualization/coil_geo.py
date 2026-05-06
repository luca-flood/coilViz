import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
h = 35  # mm
R_outer = 50  # mm

# Create radial grid
r = np.linspace(0, R_outer, 200)
theta = np.linspace(0, 2*np.pi, 200)
r, theta = np.meshgrid(r, theta)

# Parabolic profile (your equation rewritten in radial form)
z = h * (1 - (r**2) / (R_outer**2))

# Convert to Cartesian coordinates
x = r * np.cos(theta)
y = r * np.sin(theta)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none', alpha=0.9)

ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.set_title("3D Parabolic Coil Surface")

plt.show()
