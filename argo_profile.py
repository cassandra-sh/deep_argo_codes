# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 13:31:14 2018

@author: Cassandra
"""

import os
import gsw
import sys
import math
import glob
import astropy.time
import pandas as pd
import xarray as xr
import numpy  as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib
import which_ocean
import scipy
import aviso_interp
import scipy.io
from scipy import signal
from scipy.misc import derivative

REF_TIME = astropy.time.Time('1950-01-01')

def is_between(value, pair):
    """
    Figure out if a value is between two values in the pair, given that the
    pair may not be in order
    """
    if   value > pair[0] and value < pair[1]: return True
    elif value < pair[0] and value > pair[1]: return True
    else:                                     return False

def format_jd(jd):
    """
    Give a string representation of the day
    
    Gotta convert from julian day after 1950.01.01
    """
    return astropy.time.Time(jd+REF_TIME.jd, format='jd').datetime.strftime("%d %B %Y")

UNIT_DICT = {'SA'            : 'Absolute Salinity (g/kg)',
             'C'             : 'Conductivity',
             'pt'            : 'Potential Temperature (C)',
             'CT'            : 'Conservative Temperature (C)',
             'alpha'         : 'Alpha',
             'beta'          : 'Beta',
             'C_adj'         : 'Adjusted Conductivity',
             'alpha_on_beta' : 'Alpha / Beta',
             'SR'            : 'Reference Salinity',
             'z'             : 'Depth (m)',
             'integrand'     : 'Steric integrand (unitless)'}
 

SYMB_DICT = {'SA'            : 'SA',
             'C'             : 'Conductivity',
             'pt'            : r'$\theta$',
             'CT'            : 'Conservative Temperature (C)',
             'alpha'         : r'$\alpha$',
             'beta'          : r'$\beta$',
             'C_adj'         : 'Adjusted Conductivity',
             'alpha_on_beta' : r'$\frac{\alpha}{\beta}$',
             'SR'            : 'Reference Salinity',
             'z'             : 'z',
             'integrand'     : 'int'}
    
   
def attr_to_name(attr):
    return UNIT_DICT.get(attr, attr)
   
def attr_to_symb(attr):
    return SYMB_DICT.get(attr, attr)


class Profile:
    """
    Profile object stores information from an argo float profile, and can
    produce other derived values from it.

    """
    
    def __init__(self, **kwargs):
        """
        Construct this profile from inputs
        
        @**kwargs
            pressure
            psal
            temperature
            observation_number
            float_number
            julian_day
            latitude
            longitude
        """
        
        self.gen_attrlist = ['SA', 'C', 'pt', 'CT', 'C_adj', 'alpha', 'beta',
                             'alpha_on_beta', 'SR', 'z', 'integrand', 'Nsquared']
        self.list_attrlist = ['pressure', 'psal', 'temperature']
        self.single_attrlist = ['observation_number', 'float_number',
                                'datetime', 'julian_day','latitude', 'longitude']
        self.gen_single_attrlist = ['f', 'betaf']
        
        #
        # Get parameters out of kwargs
        #
        for param_name in kwargs:
            if (param_name in self.list_attrlist) or (param_name in self.single_attrlist):
                setattr(self, param_name, kwargs[param_name])
        
        #
        # Order psal and temperature by pressure
        #
        self.psal =        [ps   for _, ps   in sorted(zip(self.pressure, self.psal))]
        self.temperature = [temp for _, temp in sorted(zip(self.pressure, self.temperature))]
        self.pressure = sorted(self.pressure)
        
        #
        # Make interpolators 
        #
        self.interp_psal = scipy.interpolate.interp1d(self.pressure, self.psal, bounds_error=False)
        self.interp_temp = scipy.interpolate.interp1d(self.pressure, self.temperature, bounds_error=False)
        
        #
        # Infer values and make interps
        #
        self.make_interps()
    
    def depth_param(self, param, value):
        """
        Given a param and a value of that param, return the [lower] depth
        associated with that param value. Nan if the value is never crossed.
        """
        pval = getattr(self, param)[-1]
        zval = self.z[-1]
        for z, p in zip(reversed(self.z), reversed(getattr(self, param))):
            if np.isnan(p) or np.isnan(pval): pass
            elif value < pval and value > p: return np.interp(value, [pval, p], [zval, z])
            elif value > pval and value < p: return np.interp(value, [pval, p], [zval, z])
            pval, zval = p, z
        return np.nan
    
    def find_matches(self, given_param, given_value, other_param):
        """
        Given one parameter and value, figure out what another pameter's value
        would be for that measurement.
        
        Returns a list [potentially length 1] of valid values, because most
        parameter measurements are not one-to-one. Requires some algorithm
        after the fact to pick the most appropriate one. 
        
        @params
            given_param - Param of starting value
            given_value - Starting value
            other_param - Other param to figure out value of
        
        @returns
            list, ordered from least to most deep, of matching (interpolated)
            other_param values
        """
        gvals, ovals = getattr(self, given_param), getattr(self, other_param)
        valid_values = []
        for i in range(1, len(self.pressure)):
            if is_between(given_value, gvals[i-1:i+1]):
                interp_oval = np.interp(given_value, gvals[i-1:i+1], ovals[i-1:i+1])
                valid_values.append(interp_oval)
        return valid_values
    
    def grid_param(self, gridded, interpd, n=1000):
        """
        Project one parameter onto a grid of another parameter.
        
        ArgoProfile handles data on an uneven pressure grid, interpolating by
        pressure. If you want to get evenly spaced data, you would input an 
        even pressure grid to the parameter interpolator objects. 
        
        Here, instead, a given param (gridded) is taken as a grid, and
        another param  (interpd) is projected onto that grid
        
        CHOSEN PARAMS MUST BE ONE-TO-ONE or else this won't work. 
        
        @params
            gridded - param name to put onto grid
            interpd - param name to get values of for each grid value
            n       - number of positions in the evenly spaced grid
                      default is 1000
            
        @returns
            grid, vals
            
            grid - grid positions, evenly spaced
            vals - value at each grid position
        """
        gridded_param_values = getattr(self, gridded)
        interpd_param_values = getattr(self, interpd)
        
        grid = np.linspace(min(gridded_param_values), max(gridded_param_values), n)
        vals = np.interp(grid, gridded_param_values, interpd_param_values)
        
        return grid, vals
    
    def infer_values(self):
        """
        Generate every possible inferrable value
        
        I recommend calling this instead of any individual value generator
        """
        for attr in self.gen_attrlist:
            if hasattr(self, attr):
                pass
            else:
                getattr(self, 'get_'+attr)()
    
        for attr in self.gen_single_attrlist:
            if hasattr(self, attr):
                pass
            else:
                getattr(self, 'get_'+attr)()
                
    def make_interps(self):
        """
        Generate interpolators for every possible inferrable value
        """
        self.infer_values()
        for attr in self.gen_attrlist:
            setattr(self, attr+'_interp', 
                    scipy.interpolate.interp1d(self.pressure, 
                                               getattr(self, attr), 
                                               bounds_error=False))
        self.depth_to_pressure = scipy.interpolate.interp1d(getattr(self, 'z'),
                                                            self.pressure, 
                                                            bounds_error=False)
        
    def drop_nan(self):
        """
        Drop all rows that have a nan value in this profile
        """
        not_nan = np.ones(len(self.pressure))
        not_nan = np.logical_and(not_nan, np.logical_not(np.isnan(self.pressure)))     
        not_nan = np.logical_and(not_nan, np.logical_not(np.isnan(self.temperature)))     
        not_nan = np.logical_and(not_nan, np.logical_not(np.isnan(self.psal)))

        good = np.where(not_nan)[0]
        
        for attr_name in self.list_attrlist:
            if hasattr(self, attr_name):
                attr = getattr(self, attr_name)
                attr = np.array([attr[i] for i in good], dtype=float)
        
        for attr_name in self.gen_attrlist:
            if hasattr(self, attr_name):
                attr = getattr(self, attr_name)
                attr = np.array([attr[i] for i in good], dtype=float)
    
    def get_julian_day(self):
        """
        
        """
        if not hasattr(self, 'julian_day'):
            self.julian_day = aviso_interp.datetime_to_argo_juld(self.datetime)
        return self.julian_day
    
    def infer_height(self):
        return 0
    
    def day_str(self):
        """
        Give a string representation of the day
        
        Gotta convert from julian day after 1950.01.01
        """
        return format_jd(self.get_julian_day())
    
    def get_C(self):
        """
        Get the conductivity
        """
        self.C = gsw.conversions.C_from_SP(self.psal,
                                           self.temperature,
                                           self.pressure)
        return self.C
    
    def get_Nsquared(self):
        """
        
        """
        for attr in ['CT', 'SA']:
            if not hasattr(self, attr):
                getattr(self, 'get_'+attr)()
        nsq_vals, nsq_pres =  gsw.stability.Nsquared(self.SA, self.CT, 
                                                     self.pressure,
                                                     lat=self.latitude)
        self.Nsquared = np.interp(self.pressure, nsq_pres, nsq_vals)
        return self.Nsquared
    
    def get_integrand(self):
        """
        Get the Steric Sea Level anomaly integrand
        """
        for attr in ['pt', 'SA', 'beta', 'alpha']:
            if not hasattr(self, attr):
                getattr(self, 'get_'+attr)()
        self.integrand = self.alpha * self.pt - self.beta * self.SA
        return self.integrand  
    
    def get_SA(self):
        """
        Get the absolute salinity
        """    
        self.SA = gsw.conversions.SA_from_SP(self.psal, self.pressure,
                                             self.longitude, self.latitude)
        return self.SA
        
    def get_CT(self):
        """
        Get the conserved temperature
        
        Requires absolute salinity to have been calculated
        """
        if not hasattr(self, 'SA'): self.get_SA()
            
        self.CT = gsw.conversions.CT_from_t(self.SA,
                                            self.temperature,
                                            self.pressure)
        return self.CT
    
    def get_z(self):
        """
        Get depth (in meters)
        """
        self.z = gsw.conversions.z_from_p(self.pressure, self.latitude)
        return self.z
    
    def get_SR(self):
        """
        Get reference salinity
        """
        self.SR = gsw.conversions.SR_from_SP(self.psal)
        return self.SR
        
    def get_alpha(self):
        """
        Get alpha
        
        Requires absolute salinity and conservative temperature to have been
        calculated already
        """
        if not hasattr(self, 'CT'): self.get_CT()
        if not hasattr(self, 'SA'): self.get_SA()
            
        self.alpha = gsw.alpha(self.SA, self.CT, self.pressure)
        return self.alpha
    
    def get_alpha_on_beta(self):
        """
        Get alpha/beta        
        
        Requires absolute salinity and conservative temperature to have been
        calculated already
        """
        if not hasattr(self, 'CT'): self.get_CT()
        if not hasattr(self, 'SA'): self.get_SA()
            
        self.alpha_on_beta = gsw.alpha_on_beta(self.SA, self.CT, self.pressure)
        return self.alpha_on_beta
    
    def get_f(self):
        """
        get the coriolis parameter
        """
        self.f = gsw.geostrophy.f(self.latitude)
        return self.f
    
    
    def get_betaf(self):
        """
        Get beta, the derivative of the coriolis parameter with regards to latitude
        """
        self.betaf = derivative(gsw.geostrophy.f, self.latitude)
        return self.betaf
    
    def get_beta(self):
        """
        Get beta.
        
        Requires absolute salinity and conservative temperature to have been
        calculated already
        """
        if not hasattr(self, 'CT'): self.get_CT()
        if not hasattr(self, 'SA'): self.get_SA()
            
        self.beta = gsw.beta(self.SA, self.CT, self.pressure)
        return self.beta
    
    def get_pt(self):
        """
        Get the potential temperature
        
        Requires absolute salinity and conservative temperature to have been
        calculated already
        """
        if not hasattr(self, 'CT'): self.get_CT()
        if not hasattr(self, 'SA'): self.get_SA()
            
        self.pt = gsw.conversions.pt_from_CT(self.SA, self.CT)
        return self.pt
    
    def get_C_adj(self):
        """
        Get an adjusted conductivity for this profile.
        
        Requires conductivity to have already been calculated
        """
        
        #
        # Generate conductivity if not already done
        #
        if not hasattr(self, 'C'):
            self.get_C()
        
        #
        # Calculate adjustment parameter
        #
        
        CPcor     = -9.5700E-8
        CPcor_new = -1.1660E-7
        CTcor     =  3.2500E-6

        a1 = (1.0 + CTcor*np.array(self.temperature) + CPcor    *np.array(self.pressure))
        b1 = (1.0 + CTcor*np.array(self.temperature) + CPcor_new*np.array(self.pressure))
        self.C_adj = self.C * a1 / b1
        return self.C_adj



def from_nc_ubu(file, N_PROF=0):
    """
    Load a Profile object out of a NetCDF file, using lunux-style filepaths
    """
    floatnum = int(file.split('/')[-1].split('_')[0].strip('R').strip('D'))
    obsnum   = int(file.split('/')[-1].split('_')[1].strip('.hdf').strip('.nc').strip('R').strip('D'))
    
    with xr.open_dataset(file) as xf:
        xf = xf.sel(N_PROF=N_PROF)
        
        pres = np.array(xf['PRES'].values,          dtype=float)
        temp = np.array(xf['TEMP'].values,          dtype=float)
        psal = np.array(xf['PSAL'].values,          dtype=float)
        
        lon  = float(xf['LONGITUDE'].values)
        lat  = float(xf['LATITUDE'].values)
        juld = xf['JULD'].values
        
        return Profile(pressure     = pres,     temperature        = temp, 
                       psal         = psal,     datetime           = juld, 
                       longitude    = lon,      latitude           = lat, 
                       float_number = floatnum, observation_number = obsnum)
 
def from_hdf_ubu(file, key='p'):
    """
    Load a Profile object out of an HDF, using lunux-style filepaths
    """
    floatnum = int(file.split('/')[-1].split('_')[0].strip('R').strip('D'))
    obsnum   = int(file.split('/')[-1].split('_')[1].strip('.hdf').strip('.nc').strip('R').strip('D'))
    
    df =  pd.read_hdf(file, key=key)
    
    pres = np.array(df['PRES'].values,          dtype=float)
    temp = np.array(df['TEMP'].values,          dtype=float)
    psal = np.array(df['PSAL'].values,          dtype=float)
    
    juld = np.array(df['JULD'].values,          dtype=float)[0]
    lon  = np.array(df['LONGITUDE'].values,     dtype=float)[0]
    lat  = np.array(df['LATITUDE'].values,      dtype=float)[0]
    
    return Profile(pressure     = pres,     temperature        = temp, 
                   psal         = psal,     julian_day         = juld, 
                   longitude    = lon,      latitude           = lat, 
                   float_number = floatnum, observation_number = obsnum)

def find(target, myList):
    for i in range(len(myList)):
        if myList[i] == target:
            return i

def get_argo_files_ubu(deep=False, natlantic=False):
    """
    Get a [float number][observation number] 2D dictionary of deep argo profiles
    """
    
    argo_type = "argo_data/"
    if deep:
        argo_type = "deep_argo_data/"
    
    directory = "/data/" + argo_type + 'nc/'
    files = glob.glob(directory+"*.nc")
    
    #od = None
    names, in_natlantic = [], []
    if natlantic:
        #od = which_ocean.OceanDecider()
        index = pd.read_csv("/home/cassandra/docs/argo/small_data/ar_index_global_prof.txt",
                            header=0, comment="#")
        names = [f.split('/')[-1].strip('.nc') for f in index['file']]
        in_natlantic = index['in_natlantic'].values #od.in_natlantic(index['longitude'], index['latitude'])
    
    floats = {}
    floatnums = []
    for f in files:
        name = f.split('/')[-1].strip('.nc')
        
        if natlantic:
            if in_natlantic[find(name, names)]:
                pass
            else:
                continue

        
        float_num = int(name.split('_')[0].strip('D').strip('R'))
        floatnums.append(float_num)
        
        obs_num   = int(name.split('_')[1].strip('D'))
        
        if floats.get(float_num, None) == None:
            floats.update({float_num:{obs_num:f}})
        else:
            floats[float_num].update({obs_num:f})
            
    return floats
       
def main(fign=1, deep=True, atlantic=False, recent=True):
    """
    Load and plot a single profile, given any requested requirements
    
    @params
        fign - figure number to pass to figure. Vary if plotting multiple plots
                at once
               
    @params/requirements
        deep - whether or not to use deep argo
        atlantic - whether or not to only plot floats in the atlantic
        recent   - whether or not to use only the latest measurement for any 
                   given float (not a guarantee that the measurement will be 
                   recent to present day)
    """
    #
    # get a dictionary of all the argo profiles
    #
    floats = get_argo_files_ubu(deep=deep)

    #
    # select one that actually goes deep, if deep is specified
    # otherwise just make sure it isn't a dud (< 50 values)
    #
    a_profile = ""
    
    print("looking for float with params deep:", deep,
          'fign:', fign, 'atlantic:', atlantic, 'now searching', end='')
    for f in floats:
        for p in floats[f]:
            print('.',end='')
            sys.stdout.flush()
            
            df = None
            with xr.open_dataset(floats[f][p], decode_times=False) as xf:
                df = xf.to_dataframe()
            
            #
            # Impose requested requirements for chosen float
            #
            good = True
            
            if recent:
                obs_nums = list(floats[f])
                max_obs_num = max(obs_nums)
                if p != max_obs_num:
                    good = False
                
            if len(df) < 50:
                good = False
                
            if deep:
                try:
                    if df['PRES'].max(skipna=True) < 2000:
                        good = False
                except Warning:
                    good = False
                    
            if atlantic:
                lon = df['LONGITUDE'].values[0]
                lat = df['LATITUDE'].values[0]
                
                if lat < 0 or lat > 50:
                    good = False
                if lon < -90 or lon > -30:
                    good = False
                if np.isnan(lon) or np.isnan(lat):
                    good = False
            else:
                lon = df['LONGITUDE'].values[0]
                lat = df['LATITUDE'].values[0]
                
                if lat > 0 and lat < 50:
                    good = False
                if lon > -90 and lon < -30:
                    good = False
                if np.isnan(lon) or np.isnan(lat):
                    good = False
                
            
            #
            # If this float meets the requirements, move forward
            #
            if good:
                a_profile = floats[f][p]
                break
        if a_profile != "":
            break
    print("\nPlotting", a_profile)
                  
    
    #
    # Generate the profile object and get the conservative temp and conductivity
    #
    prof = from_nc_ubu(a_profile)
    prof.drop_nan()
    prof.infer_values()
    
    #
    # Plot the profile up
    #
    fig, ((ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8)) = plt.subplots(nrows=3, ncols=3, num=fign)    
    
    ax0.scatter(prof.pt, prof.pressure, s=5, color='black')
    ax0.set_xlabel('Potential Temperature')
    ax0.set_ylabel('Pressure')
    ax0.invert_yaxis()
    ax0.grid(True)
    
    ax1.scatter(prof.alpha_on_beta, prof.pressure, s=5, color='black')
    ax1.set_xlabel('In-Situ tmperature')
    ax1.set_ylabel('Pressure')
    ax1.invert_yaxis()
    ax1.grid(True)
    
    ax2.scatter(prof.SA - prof.psal, prof.pressure, s=5, color='black')
    ax2.set_xlabel("Absolute Salinity minus Practical Salinity")
    ax2.set_ylabel("Pressure")    
    ax2.invert_yaxis()
    ax2.grid(True)
    
    ax3.scatter(prof.beta, prof.pressure, s=5, color='black')
    ax3.set_xlabel("Beta")
    ax3.set_ylabel("Pressure")    
    ax3.invert_yaxis()
    ax3.grid(True)
    
    ax4.scatter(prof.alpha, prof.pressure, s=5, color='black')
    ax4.set_xlabel('Alpha')
    ax4.set_ylabel('Pressure')
    ax4.invert_yaxis()
    ax4.grid(True)
    
    ax5.scatter(prof.alpha, prof.beta, s=5, color='black')
    ax5.set_xlabel('Alpha')
    ax5.set_ylabel('Beta')
    ax5.invert_yaxis()
    ax5.grid(True)
    
    ax6.scatter(prof.alpha_on_beta, prof.pressure, s=5, color='black')
    ax6.set_xlabel('Alpha/Beta')
    ax6.set_ylabel('Pressure')
    ax6.invert_yaxis()
    ax6.grid(True)
    
    ax7.scatter(prof.CT - prof.pt, prof.pressure, s=5, color='black')
    ax7.set_xlabel('Conservative minus Potential Temperature')    
    ax7.set_ylabel('Pressure')
    ax7.invert_yaxis()
    ax7.grid(True)
    
    
    #
    # Plot location on map
    #
    plt.sca(ax8)
    map = Basemap(llcrnrlat = prof.latitude[0] - 30.0, urcrnrlat = prof.latitude[0]  + 30.0,
                  llcrnrlon = prof.longitude[0]- 40.0, urcrnrlon = prof.longitude[0] + 40.0)
    map.drawmapboundary()
    map.fillcontinents()
    map.drawcoastlines()
    x,y = map(prof.longitude[0], prof.latitude[0])
    parallels = np.arange(-80.,81,10.)
    map.drawparallels(parallels, labels=[False, True, False, False])
    meridians = np.arange(10.,351.,20.)
    map.drawmeridians(meridians, labels=[False, False, False, True])
    map.scatter(x, y, color='black', zorder=1, s=60, marker='+')
    ax8.set_title("Location")
    
    #
    # Show profile number, float number, date at top
    #
    fig.suptitle(("Profile for " +
                  a_profile.split('/')[-1].strip('.hdf').strip('.nc') +
                  " on " + prof.day_str()))
    
        
    

if __name__ == "__main__":
    font = {'size'   : 12}
    matplotlib.rc('font', **font)

    main(deep=True,  fign=1, atlantic=False)
    #main(deep=True,  fign=2, atlantic=True)
    #main(deep=False, fign=3, atlantic=False)
    #main(deep=False, fign=4, atlantic=True)
    
    plt.show()