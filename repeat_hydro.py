#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 19:13:31 2018

@author: cassandra
"""
import scipy.io
from scipy import signal
import gsw
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
import math
import numpy as np


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx
    
class RepeatHydrography:
    """
    
    """
    
    def __init__(self, **kwargs):
        """
        
        """
        self.hydro_dir = kwargs.get('repeat_hydrography_dir', "/data/repeat_hydrography/ctd_all_gridded_A05.mat")
        self.hydro_dct = scipy.io.loadmat(self.hydro_dir)
        
        self.hydro_longitudes = self.hydro_dct['l_grid'][0]
        self.hydro_pressure   = self.hydro_dct['pr_grid'][0]
        self.n_repeats = len(self.hydro_dct['D_pr']['th'][0])
        
        # Quick and dirty - get the latest repeat hydrography profile
        # longitude, pressure indexing (after transposing)
        
        self.hydro_pt = self.hydro_dct['D_pr']['th'][0][self.n_repeats - 1].T
        self.hydro_latitudes     = self.hydro_dct['D_pr']['lat'][0][self.n_repeats - 1][0]
        self.max_pressures = self.hydro_dct['D_pr']['maxpr'][0][self.n_repeats - 1][0]
        self.zvals = [self.get_z(lon) for lon in self.hydro_longitudes]
    
        smooth_axis = kwargs.get("smooth_axis", "longitude")
        if smooth_axis == 'pressure':
            self.hydro_pt = signal.savgol_filter(self.hydro_pt, 7, 3, axis=1)
        elif smooth_axis == 'longitude':
            self.hydro_pt = signal.savgol_filter(self.hydro_pt, 51, 3, axis=0)
    
    def get_hydro_pt(self, lon):
        """
        Get the repeat hydrography potential temperature profile for a given logitude 
        """
        return self.hydro_pt[find_nearest(self.hydro_longitudes, lon)]
    
    def get_lat(self, lon):
        """
        Get the repeat hydrography latitude for a given logitude 
        """
        return self.hydro_latitudes[find_nearest(self.hydro_longitudes, lon)]
    
    def get_z(self, lon):
        """
        Get the depth grid associated with the given longitude's measurements 
        """
        return gsw.conversions.z_from_p(self.hydro_pressure, self.get_lat(lon))
    
    def get_hydro_isopycnal_heights(self, lon, pt_grid=None):
        """
        Get the isopycnal heights from the potential temperature profile of the repeat hydrography
        associated with the longitude given
        """
        pt = self.get_hydro_pt(lon)   
        if pt_grid is None:
            pt_grid = np.arange(1.0, 30.0, 0.1)        
        return pt_grid, np.interp(pt_grid, np.flip(pt, 0), np.flip(self.get_z(lon), 0))
    
    def axplot_isopycnals(self, axis):
        """
        
        """
        axis.contourf(self.hydro_longitudes, self.hydro_pressure,
                      self.hydro_pt.T, 50, cbar='ocean')
        axis.contour(self.hydro_longitudes, self.hydro_pressure,
                     self.hydro_pt.T, np.linspace(1,2,num=10), colors='white')
        axis.fill_between(self.hydro_longitudes, 7000, self.max_pressures, color='black')
        axis.invert_yaxis()
        axis.set_xlabel("Repeat Hydrography Longitude (degrees)")
        axis.set_ylabel("Pressure (dbar)")
        

def main():
    ReHydro = RepeatHydrography(smooth_axis='pressure')
    plt.figure(1)
    axis = plt.subplot()
    ReHydro.axplot_isopycnals(axis)
    plt.show()
    
if __name__ == "__main__":
    main()
    