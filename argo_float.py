#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:56:58 2018

@author: cassandra
"""
import argo_profile
import aviso_interp
import glob
import time
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import sys
import gc
import os
import numpy as np

def ensure_dir(file_path):
    """
    Ensure directory exists
    @author Parand via stackoverflow
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def num_only(string):
    return "".join(_ for _ in string if _ in "1234567890")

class ArgoFloat:
    """
    Contains all profiles for a given argo float, and an aviso interpolator
    clipped to the location of the float.
    """
    def __init__(self, WMO_ID, **kwargs):
        """
        
        @params 
            WMO_ID
        **kwargs
            argo_dir
            lim
            
        """
        
        # STEP 1. GET THE PROFILES
        
        self.argo_dir     = kwargs.get('argo_dir', "/data/argo_data/nc/")      # Get the directory
        self.WMO_ID       = WMO_ID                                             # and using the float number
        obs_files         = glob.glob(self.argo_dir+"*"+str(WMO_ID)+"*.nc")    # get all relevant files.
        self.obs_nums = [int(num_only(f.split('_')[-1])) for f in obs_files]   # Get observation numbers from
                                                                               # the file names.
        obs_files = [f for _,f in sorted(zip(self.obs_nums, obs_files))]       # Sort by observation number.
        self.obs_nums = sorted(self.obs_nums)                                
        if self.obs_nums[0] == 0: obs_files.pop(0), self.obs_nums.pop(0)       # Get rid of 000 profiles
        lim = kwargs.get('lim', 30)                                            # Get a number of profiles up 
        obs_files = obs_files[:lim]                                            # to some limit
        self.obs_nums = self.obs_nums[:lim]
        self.profiles = [argo_profile.from_nc_ubu(f) for f in obs_files]       # Get the profiles themselves
        self.julds    = [prof.get_julian_day() for prof in self.profiles]      # And associated julian days
        
        
        # STEP 2. GET THE AVERAGE PROFILE
        
        # Get the average location
        self.lon_avg = np.sum([prof.longitude for prof in self.profiles])/len(self.profiles)
        self.lat_avg = np.sum([prof.latitude  for prof in self.profiles])/len(self.profiles)
        
        # Get the average P/T/S profile
        self.pres_avg = np.arange(0, 6000.05, 0.5)
        psal_interps = np.array([prof.interp_psal(self.pres_avg) for prof in self.profiles])
        temp_interps = np.array([prof.interp_temp(self.pres_avg) for prof in self.profiles])
        
        psal_avg = np.nanmean(psal_interps, axis=0)  # take the average of all interpolated
        temp_avg = np.nanmean(temp_interps, axis=0)  # values which are NOT nan (the fill value)
                                                     # i.e. where there IS data
        
        # Make the average profile object
        self.prof_avg = argo_profile.Profile(temperature = temp_avg,
                                             pressure    = self.pres_avg,
                                             psal        = psal_avg,
                                             longitude   = self.lon_avg,
                                             latitude    = self.lat_avg)
        
        
        # STEP 3. GET INTERPOLATORS
        
        # Make all profle interpolators
        for prof in self.profiles:
            prof.make_interps()
        self.prof_avg.make_interps()
        
        # Get an aviso interpolator within a degree range of this location
        self.AIbox = [self.lon_avg-10.0, self.lon_avg+10.0,
                      self.lat_avg-10.0, self.lat_avg+10.0]
        self.AI = aviso_interp.AvisoInterpolator(box=self.AIbox, irregular=False,
                                                 limit=64, verbose=False)
    
    
    def aviso_map_plot(self, date, axis):
        """
        Plot the sea level anomaly for a given day on a given axis
        """
        # 1. Plot the basemap
        plt.sca(axis)
        map = Basemap(llcrnrlat = self.lat_avg - 10.0, urcrnrlat = self.lat_avg + 10.0,
                      llcrnrlon = self.lon_avg - 10.0, urcrnrlon = self.lon_avg + 10.0)
        map.drawmapboundary()
        map.fillcontinents()
        map.drawcoastlines()
        parallels = np.arange(-80.,81,10.)
        map.drawparallels(parallels, labels=[False, True, False, False])
        meridians = np.arange(10.,351.,20.)
        map.drawmeridians(meridians, labels=[False, False, False, True])
        
        
        # 2. Interpolate to this date
        lons = np.arange(self.AIbox[0], self.AIbox[1], 0.05)       # Get a range of lat/lon values
        lats = np.arange(self.AIbox[2], self.AIbox[3], 0.05)
        lons[lons<0] = lons[lons<0]+360                            # Fix the lons to be from 0 to 360
        dd, la, lo = np.meshgrid(date, lats, lons, indexing='ij')  # Turn these into a grid
        vals = self.AI.interpolate(dd, la, lo)                     # Interpolate on the grid
        
        
        # 3. Plot the sea level anomaly
        map.pcolor(lons, lats, vals[0], cmap='coolwarm',           
                   latlon=True, vmin=self.AI.min, vmax=self.AI.max)
        cbar = plt.colorbar(orientation='horizontal')
        cbar.set_label('Sea Level Anomaly (m)')
        
        
        # 4. Return map
        return map
    
    def aviso_sla(self, prof_index):
        """
        Get the aviso interpolated sea level anomaly for a given profile index
        """
        lat = self.profiles[prof_index].latitude
        lon = self.profiles[prof_index].longitude
        day = self.profiles[prof_index].get_julian_day()
        if lon < 0: lon = lon + 360
        return self.AI.interpolate(day, lat, lon)
    
    def steric_sla(self, prof_index):
        """
        Calculate steric sea level anomaly for the profile specified by the 
        given prof index.
        
        This is relative to average SLA across all available float values
        """
        # Compare this profile to the average and get the differential between
        # potential temperature and absolute salinity
        delta_pt = self.profiles[prof_index].pt_interp(self.pres_avg) - self.prof_avg.pt
        delta_SA = self.profiles[prof_index].SA_interp(self.pres_avg) - self.prof_avg.SA
        
        # Get the average profile's beta and alpha values
        beta  = self.prof_avg.beta
        alpha = self.prof_avg.alpha
        
        # comprise integrand
        integrand = alpha*delta_pt - beta*delta_SA  
        
        # Drop nan values - effectively only going as deep as the deepest measurement
        int_pressure = self.pres_avg[~np.isnan(integrand)]
        integrand = integrand[~np.isnan(integrand)]
        
        # Integrate
        return np.trapz(integrand, x=int_pressure)  # integrate over pressure
        
        
    
    def plot_profiles(self, prof_indices='all', saveas=None):
        """
        Make a figure depicing information on a given profile, plus profiles
        coming before it.
        
        @params
            prof_indices - list of indices for self.profiles to plot. Will plot
                           information sequentially from beginning to end, and
                           will take the last one's time to plot the 
                           interpolated sea level anomaly.
                        
                           if 'all' - plot all profiles and take the last one's 
                           date. 
        """
        # figure out prof nums
        if prof_indices == 'all': prof_indices = list(range(len(self.profiles)))
            
        # Plan not to display if saving
        if saveas != None:  plt.ioff()
        else:               plt.ion()
            
        # Get figure and axes
        f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 7))
        
        # Plot all axis labels, adjust size and shape for ~ok~ saved formatting
        ax1.set_xlabel('Julian Day'), ax1.set_ylabel("Sea Level Anomaly (m)")
        ax2.set_ylabel("Pressure"),   ax2.set_xlabel("Temperature (in-situ)")
        plt.tight_layout(), plt.subplots_adjust(bottom=0.3, left=0.1)
        
        # Get profiles information
        lats       = [self.profiles[i].latitude         for i in prof_indices]
        lons       = [self.profiles[i].longitude        for i in prof_indices]
        days       = [self.profiles[i].get_julian_day() for i in prof_indices]
        aviso_sla  = [self.aviso_sla(i)                 for i in prof_indices]
        steric_sla = [self.steric_sla(i)                for i in prof_indices]
        
        
        
        # AXIS 1 : Float Sea Level Anomaly vs. Time
        ax1.plot(days, aviso_sla,  label='Aviso')             # Plot the aviso SLA
        ax1.plot(days, steric_sla, label='Steric')            # Plot the steric SLA
        xticks = ax1.get_xticks()                             # Get the automatic xticks
        form = [argo_profile.format_jd(xt) for xt in xticks]  # Convert into formatted dates
        ax1.set_xticklabels(form, rotation=90)                # Use dates as tick labels
        ax1.grid(True)                                        # Add grid
        ax1.legend(loc='lower right')                         # Add legend
        
        
        # AXIS 2 : Pressure vs. Temperature for floats
        for i in prof_indices:
            ax2.scatter(self.profiles[i].temperature,         # Plot the P/T profiles
                        self.profiles[i].pressure, s=1)       # Use very small markers
        ax2.invert_yaxis()                                    # Invert pressure axis to descending
        ax2.grid(True)                                        # Add grid
        
        
        # AXIS 3 : Location over time
        map = self.aviso_map_plot(days[-1], ax3)              # Plot the interpolated SLA values
        x, y = map(lons, lats)                                # Get map float locations
        map.plot(x,y, label='Float location', color='black')  # Plot float locations
        
        
        # Save and close if desired
        if saveas != None:
            f.savefig(saveas)
            plt.close(f)
        else:
            plt.show()
    
    def plot_prof_all(self,
                      savedir='/home/cassandra/docs/argo/movies/float/all/',
                      WMO_ID_dir=True):
        """
        Plot all of the profiles for this float
        """
        # If float_num_dir, modify savedir for the float number
        if WMO_ID_dir: savedir = savedir+str(self.WMO_ID)+'/'
        
        # Make the save directory if it does not exist
        ensure_dir(savedir)
        
        # Get all indices and interate through them
        indices = list(range(1, len(self.profiles)))
        for i in indices:
            
            # Report progress
            print('Plotting index', i)
            
            # Plot profiles from beginning to this index
            self.plot_profiles(prof_indices = indices[:i+1],
                               saveas       = savedir+'f{:0>3d}.png'.format(i))
            
    def plot_prof_relevant(self,
                           savedir='/home/cassandra/docs/argo/movies/float/rel/',
                           WMO_ID_dir=True):
        """
        Plot all the profiles which have aviso data overlap
        """
        # If float_num_dir, modify savedir for the float number
        if WMO_ID_dir: savedir = savedir+str(self.WMO_ID)+'/'
        
        # Make the save directory if it does not exist
        ensure_dir(savedir)
        
        # Get the earliest and latest aviso days, for the range of interpolate-able values
        earliest_day = min(self.AI.days)
        latest_day   = max(self.AI.days)
        
        # Figure out where the profile data falls between these days
        date_ok = np.logical_and(np.greater(self.julds, earliest_day),
                                 np.less(self.julds, latest_day))
        
        # Get as list of acceptable profile indices
        where_date_ok = np.where(date_ok)[0].tolist()
        
        # Get first valid date index
        first = where_date_ok.pop(0)
        
        # Plan to use the acceptable indices as a slice of the array of possible
        # indices, only plotting profiles from the first valid profile and up
        indices = list(range(len(self.profiles)))
        for i in where_date_ok:
            indices_to_use = indices[first:i+1]
            
            # Report progress
            print('WMO-ID', self.WMO_ID, 'Plotting', first, 'to', i)
            sys.stdout.flush()
            
            # plot selected profiles
            self.plot_profiles(prof_indices = indices_to_use, 
                               saveas       = (savedir+'f{:0>3d}.png'.format(i)))




def main():
    """
    Make movies for every deep argo float in the northwest atlantic
    """    
    for WMO_ID in [4902321, 4902322, 4902323, 4902324, 4902325, 4902326]:
        ArgoFloat(WMO_ID, argo_dir="/data/deep_argo_data/nc/").plot_prof_relevant()
        gc.collect()
        
if __name__ == '__main__':
    main()