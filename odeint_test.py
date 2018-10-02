#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:52:35 2018

@author: cassandra
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import interpolate
import argo_float
from matplotlib import rc
import matplotlib
from scipy.optimize import minimize

af_weekly = argo_float.ArgoFloat(4902323, argo_dir="/data/deep_argo_data/nc/",
                                 aviso_dir = '/data/aviso_data/nrt/weekly/')

prof_n = 10

NN = af_weekly.profiles[prof_n].Nsquared
zN = af_weekly.profiles[prof_n].z

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

NsqInterp = interpolate.interp1d(zN, NN, bounds_error=False)
    
zInc = -.1
maxd = af_weekly.profiles[prof_n].z[-1]
zgrid = np.arange(zN[0], maxd, zInc)

zgridN = NsqInterp(zgrid)
zgridN = smooth(zgridN, abs(int(1.0/zInc)))
N_avg = np.nanmean(zgridN)

Interp = interpolate.interp1d(zgrid, zgridN/N_avg, bounds_error=False)


f = af_weekly.profiles[prof_n].f
beta = af_weekly.profiles[prof_n].betaf

k1 = 1 #f**4 / (beta**2)
norm = (np.pi**2) / (maxd**2)

# Heave calc
hh, zh = af_weekly.calc_isopycnal_heave(prof_n)

m_max = 20
    
 


def func(y, z, params):
    F, Fp = y
    k, NsqInterp = params
    N = NsqInterp(z)
    derivs = [Fp, -1.0*(k*N)*F]
    return derivs
       
def zeros(y):
    ii = []
    for i in range(len(y)-1):
        if y[i] == 0:
            ii.append(i)
        elif y[i] > 0 and y[i+1] < 0:
            ii.append(i)
        elif y[i] < 0 and y[i+1] > 0:
            ii.append(i)
        elif y[i+1] == 0:
            ii.append(i+1)
    return ii

def bounds_check(m, y):
    """
    Assuming y[0] is always 0, try to give a goodness parameter for how close
    to satisfying the boundary conditions we are
    """
    iz = zeros(y)    
    z_diff = abs(len(iz) - (m+1))
    bound_diff = abs(iz[-1] - len(y))/len(y)
    return z_diff + bound_diff     

def model(k, NsqInterp, zgrid):
    psoln, idict = odeint(func, [0, 0.1], zgrid, args=([k, NsqInterp],), full_output=1, printmessg=1,hmax=1.0)    
    return psoln[:,0]

def check_model(k, params):
    NsqInterp, zgrid, m = params
    check = bounds_check(m, model(k, NsqInterp, zgrid))
    return check

def pseudo_min(k0, NsqInterp, zgrid, m, bounds, n_iter = 4):
    params=[NsqInterp, zgrid, m]
    n_remaining = n_iter
    n_range = bounds
    checks = []
    while n_remaining > 0:
        krange = np.linspace(n_range[0], n_range[1], num=10)
        n_remaining = n_remaining - 1
        checks = [check_model(k, params) for k in krange]
        least = np.argsort(checks)[0]
        
        if least == 0:
            least = 1
        elif least == len(checks)-1:
            least = len(checks)-2
        
        k_best = [krange[least-1], krange[least+1]]
        n_range = [k_best[0], k_best[-1]]
        
        print("")
        print("Iterations remaining: " + str(n_remaining))
        for i in range(len(krange)):
            if i == least or i == least + 1 or i == least - 1:
                print("~~", end='')
            print(krange[i], checks[i])
        print("")
        print("new range: ", n_range)
        print("")
        
    return np.linspace(n_range[0], n_range[1], num=10)[np.argmin(checks)]
        

def solve_model(k0, NsqInterp, zgrid, m):
    return pseudo_min(k0, NsqInterp, zgrid, m, [k0/16, k0])

# Prepare to plot results
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
rc('text', usetex=True)
rc('font', family='serif', size=18)
fig = plt.figure(0, figsize=(13,8))
ax1 = fig.add_subplot(121)
ax3 = fig.add_subplot(122)

kr_list = []

for m in range(1, m_max+1):
    
    print("")
    print("")
    print("Calculating mode = " + str(m))
    
    k0 = 4 * m**2 * k1 * norm   # Hazard a first guess of a k value

    kr = solve_model(k0, Interp, zgrid, m)  # Send to the solver
    
    kr_list.append(kr)
    
    # Generate the solved model and plot
    psoln, idict = odeint(func, [0, 0.1], zgrid, args=([kr, Interp],), full_output=1, printmessg=1,hmax=1.0)   
    
    plt.figure(0)
    # Plot results
    ax1.plot(psoln[:,0], zgrid, label='m = '+str(m))
    ax1.set_xlabel('Isopycnal Heave')
    ax1.set_ylabel('Depth (m)')
    
    # Plot coefficient
    ax3.plot(kr*Interp(zgrid), zgrid, label='m = '+str(m))
    ax3.set_xlabel('Coefficient')
    ax3.set_ylabel("Depth (m)")
    
ax1.legend()
ax1.grid(True)
ax3.legend() 
ax3.grid(True)   
plt.tight_layout()
plt.show()

print("")
print("")
print("Final kr values:")
for m in range(0,m_max):
    print(m+1, kr_list[m])
    
    
kr = [5.513858166966546e-07,  2.582267493198889e-06,  4.903927911759009e-06,
      8.529209950976794e-06,  1.4147034688720004e-05, 2.0371729951756807e-05,
      3.0052839876674284e-05, 3.746535682468868e-05,  4.78008835503735e-05,
      5.753577164161308e-05,  7.244114553254764e-05,  8.612999972053219e-05,  
      0.00010003129232167567, 0.00011508394281862547, 0.00013278023208437083,
      0.00015190702940815002, 0.0001714887949177944,  0.0001987167371437238,
      0.0002242435819685213,  0.0002413701270380864]