import MDAnalysis as mda
import sys
import numpy as np
from MDAnalysis.lib.distances import distance_array, calc_angles, calc_bonds
from MDAnalysis.analysis import lineardensity

from scipy.spatial import distance
from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt

import pandas as pd
from multiprocessing import current_process, Pool
import multiprocessing
import os


u=mda.Universe("./simulation.pos_0.xyz",dt=0.0005)
#u=mda.Universe("/users/jglan/jglan_s1218/jglan/uzh1_result/pimd_ionization/md_traj/pimd/open-piglet-cpu/simulation.xc.pdb",dt=0.0005)

u.dimensions=[14.240302648, 16.443320454, 50.0, 90, 90, 90]
cutoff=1.32
n=0
iter=0

cn_d2o=np.zeros(u.trajectory.n_frames*134*2)
z_d2o=np.zeros(u.trajectory.n_frames*134*2)

for ts in u.trajectory[::10]:
    oxygen = u.select_atoms('name O')
    hydrogen = u.select_atoms('name H')
    d_int=distance_array(oxygen.positions,hydrogen.positions)
    cn_int= (1-(d_int/cutoff)**16) / ((1-(d_int/cutoff)**56))
    #oh1 = np.diagonal(cn_int[:,::2])
    #oh2 = np.diagonal(cn_int[:,1::2])
    cn_d2o[134*iter: 134*iter + 134] = np.sum(cn_int,axis=1)
    z_d2o[134*iter: 134*iter + 134] = oxygen.positions[:,2]
    iter = iter + 1 
    #print(z,cn)




#plt.plot(all-32)
plt.hist2d(z_d2o[np.where(z_d2o!=0)],cn_d2o[np.where(z_d2o!=0)],bins=500,cmin = 2)
plt.show()
