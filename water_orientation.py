import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt
import matplotlib as mpl


def load_traj(traj):
    cell = [14.240302648, 16.443320454, 50.0, 90, 90, 90]
    u = mda.Universe(traj)
    u.dimensions = cell
    return u, cell


# Load your trajectory and topology files
u, cell = load_traj("./simulation.pos_0.xyz")

# Get positions of the oxygen and hydrogen atoms
oxygen =  u.select_atoms("name O", updating=True)
hydrogen1 = u.select_atoms("name H", updating=True)[0::2]
hydrogen2 = u.select_atoms("name H", updating=True)[1::2]

# Number of frames and water molecules
n_frames = len(u.trajectory)
n_waters = len(oxygen)

# Initialize arrays for dipole cosines and z positions
dipole_vectors = np.zeros((n_frames, n_waters))
z_positions = np.zeros((n_frames, n_waters))

# Calculate the dipole vector for each water molecule
for i, ts in enumerate(u.trajectory):
    O_pos = oxygen.positions
    H1_pos = hydrogen1.positions
    H2_pos = hydrogen2.positions

    dipole = (H1_pos + H2_pos) / 2 - O_pos
    dipole_magnitudes = np.linalg.norm(dipole, axis=1)
    dipole_unit_vectors = dipole / dipole_magnitudes[:, np.newaxis]
    
    dipole_vectors[i] = dipole_unit_vectors[:, 2]  # z-component
    z_positions[i] = O_pos[:, 2]

# Flatten the arrays
dipole_vectors = dipole_vectors.flatten()
z_positions = z_positions.flatten()

# Plot the 2D histogram
plt.hist2d(z_positions, dipole_vectors,range=([8,12],[-1,1]),bins=300,norm=mpl.colors.LogNorm(),cmap=plt.cm.jet)
plt.colorbar(label='Density')
plt.xlabel('Z')
plt.ylabel('cos(theta)')
plt.title('Water Dipole Orientation w.r.t Z-axis')
plt.show()
