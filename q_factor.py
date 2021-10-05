import MDAnalysis as mda
from MDAnalysis.analysis import rdf
import matplotlib.pyplot as plt
import numpy as np
from MDAnalysis.lib.distances import distance_array, calc_angles, calc_bonds
import math


def get_cell(traj):
    au2ag = 0.529177249
    with open (traj, 'r') as infile:
        for line in infile:
            if "CELL(abcABC)" in line:
                if "cell{atomic_unit}" in line:
                    cell = np.asarray(line.split()[2:8],dtype=np.float64)
                    cell[0:3] = cell[0:3] * au2ag
                if "cell{angstrom}" in line:
                    cell = np.asarray(line.split()[2:8],dtype=np.float64)
                break
    return cell

def load_traj(traj):
    cell = get_cell(traj)
    u = mda.Universe(traj,box=cell,dimensions=cell,dt=0.5)
    u.dimensions=cell

    return u,cell
    
def cal_rdf(a,b,traj):
    u,cell = load_traj(traj)
    ele_a = u.select_atoms('name '+str(a),periodic=True)
    ele_b = u.select_atoms('name '+str(b),periodic=True)
    gab = rdf.InterRDF(ele_a,ele_b, nbins=200,range=(0, cell[0]/2), density=True,exclusion_block=(1, 1))
    gab.run(start=20000,step=50)
    outfile = './g'+str(a)+str(b)+'.dat'

    np.savetxt("g"+str(a)+str(b)+".dat", np.transpose([gab.bins, gab.rdf]), fmt="%s")
    return gab.bins, gab.rdf

    
u,cell = load_traj("./simulation.pos_0.xyz")

O0 = u.select_atoms('name O')
dis_o = distance_array(O0.positions,O0.positions,box=cell)

theta = np.zeros(dis_o.shape[0]*10000*6*128)
cos = np.zeros(dis_o.shape[0]*10000*6*128)


t=0
for ts in u.trajectory[10001:-1:1000]:
    O0 = u.select_atoms('name O')
    dis_o = distance_array(O0.positions,O0.positions,box=cell)
    for idx in range(dis_o.shape[0]):
        test_list = dis_o[idx,:]
        #print(len(test_list[test_list < 3.25]))
        K = len(test_list[test_list < 3.35])
        res = sorted(range(len(test_list)), key = lambda sub: test_list[sub])[:K]
        #print(res)
        for i in range(K-1):
            for j in range(K-i-2):
                #print(res[i+1],res[j+2])
                a,b,c = dis_o[res[0],res[i+1]], dis_o[res[0],res[i+j+2]], dis_o[res[i+1],res[i+j+2]]
                cos[t] =  (a**2 + b**2 - c**2 ) / (2 * a * b)
                theta[t]= math.acos(cos[t])/math.pi*180
                t=t+1
                #print(theta)
                
    if ts.frame % 1000 == 0 :
        print(ts.frame)    

q_factor = 1 - np.average((cos[cos !=0] + 1/3)**2)*18/8
        
#ooo=np.genfromtxt('./ooo.txt')
hist = plt.hist(theta[theta !=  0],bins=180,density=True)
np.savetxt("avg_q.dat", [q_factor,])
np.savetxt("q_factor.dat", np.transpose([hist[1][1:],hist[0]]), fmt="%s")
#plt.plot(ooo[:,0],ooo[:,1])
#plt.show()
np.savetxt("avg_q.dat", [q_factor,])

np.savetxt("q_factor.dat", np.transpose([hist[1][1:],hist[0]]), fmt="%s")
q_factor = 1 - np.average((cos[cos !=0] + 1/3)**2)*18/8
