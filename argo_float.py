#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:56:58 2018

@author: cassandra
"""

from mpl_toolkits.basemap  import Basemap
from matplotlib            import rc
from matplotlib.pyplot     import cm
import matplotlib.gridspec as gridspec
import matplotlib
import scipy.interpolate
import matplotlib.pyplot   as plt
import numpy               as np
import aviso_interp
import argo_profile
import glob
import time
import sys
import gc
import os

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
        lim = kwargs.get('lim', None)                                          # Get a number of profiles up 
        if lim != None:  obs_files = obs_files[:lim]                           # to some limit, if provided
        self.obs_nums = self.obs_nums[:lim]
        self.profiles = [argo_profile.from_nc_ubu(f) for f in obs_files]       # Get the profiles themselves
        self.julds    = [prof.get_julian_day() for prof in self.profiles]      # And associated julian days
      
        # STEP 2. GET THE AVERAGE PROFILE
        # Get the average location
        self.lons = [prof.longitude for prof in self.profiles]
        self.lats = [prof.latitude  for prof in self.profiles]
        self.lon_avg = np.sum(self.lons)/len(self.profiles)
        self.lat_avg = np.sum(self.lats)/len(self.profiles)
        # Get the average P/T/S profile
        self.pres_avg = np.arange(0, 6001.0, 1.0)
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
        self.depth_avg = -1.0*self.prof_avg.get_z()
    
        # STEP 3. GET INTERPOLATORS
        # Make all profle interpolators
        for prof in self.profiles:
            prof.make_interps()
        self.prof_avg.make_interps()
        # Get an aviso interpolator within a degree range of this location
        self.AIbox = [self.lon_avg-10.0, self.lon_avg+10.0,
                      self.lat_avg-10.0, self.lat_avg+10.0]
        aviso_dir = kwargs.get('aviso_dir', '/data/aviso_data/nrt/weekly/')
        self.AI = aviso_interp.AvisoInterpolator(box=self.AIbox, irregular=False,
                                                 limit=100, verbose=False,
                                                 aviso_dir = aviso_dir,
                                                 units = 'm')
        # Make time profile interpolators
        temps = np.array([prof.interp_temp(self.pres_avg) for prof in self.profiles])
        psals = np.array([prof.interp_psal(self.pres_avg) for prof in self.profiles])
        self.temp_interp = scipy.interpolate.RegularGridInterpolator((self.julds, self.pres_avg), 
                                                                     temps, 
                                                                     fill_value=np.nan, 
                                                                     bounds_error=False)
        self.psal_interp = scipy.interpolate.RegularGridInterpolator((self.julds, self.pres_avg), 
                                                                     psals, 
                                                                     fill_value=np.nan, 
                                                                     bounds_error=False)
        self.lon_interp = scipy.interpolate.interp1d(self.julds, self.lons, fill_value=np.nan, bounds_error=False)
        self.lat_interp = scipy.interpolate.interp1d(self.julds, self.lats, fill_value=np.nan, bounds_error=False)
    
    def day_profile(self, day):
        """
        Interpolate from the set of argo profiles a new profile for a given day
        on the track of available argo profiles. 
        """
        return argo_profile.Profile(temperature = self.temp_interp((day, self.pres_avg)),
                                    pressure    = self.pres_avg,
                                    psal        = self.psal_interp((day, self.pres_avg)),
                                    longitude   = self.lon_interp(day),
                                    latitude    = self.lat_interp(day))
    
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
        map.drawparallels(parallels, labels=[True, False, False, False])
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
    
    def steric_sla(self, prof_index, return_integrand=False, return_deltas=False):
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
        int_depth = self.depth_avg[~np.isnan(integrand)]
        integrand = integrand[~np.isnan(integrand)]
        # Return integrand or delta_pt and delta_SA if desired
        if return_integrand:   return int_depth, integrand
        elif return_deltas:    return delta_pt, delta_SA
        # Otherwise integrate
        return np.trapz(integrand, x=int_depth)  # integrate over pressure
        
        
    
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
            
        # Get profiles information
        lats       = [self.profiles[i].latitude         for i in prof_indices]
        lons       = [self.profiles[i].longitude        for i in prof_indices]
        days       = [self.julds[i]                     for i in prof_indices]
        
        aviso_sla  = np.array([self.aviso_sla(i)  for i in prof_indices])
        steric_sla = np.array([self.steric_sla(i) for i in prof_indices])
        
        # Convert to relative sea level (to first measurement)
        aviso_sla = aviso_sla - aviso_sla[0]
        steric_sla = steric_sla - steric_sla[0]
        
        
        # Make formatted strings for axis labels
        rc('text', usetex=True)
        rc('font', family='serif')
        #steric_int_label = r"Steric Integrand ($\alpha \Delta T_{\textrm{potential}}}-\beta \Delta S_{{\textrm{absolute anomaly}}}$)"
        steric_int_label = 'Steric Integrand'

        # Get a color scheme so that the latest profiles plotted are darker.
        cmap = cm.get_cmap('binary')
        norm = matplotlib.colors.PowerNorm(vmin=1, vmax=5, gamma=3.0, clip=True)
        colors = [cmap(0.05+0.95*norm(i+1-len(prof_indices)+5)) for i in range(len(prof_indices))]

        # Get figure and axes, label with WMO_id and latest date
        f = plt.figure(figsize=(13,8))
        gs = gridspec.GridSpec(6, 9, figure=f, wspace=0.9, hspace=0.6)
        map_ax = plt.subplot(gs[ :-3   ,  0:2])
        SLA_ax = plt.subplot(gs[3:     ,  0:2])
        top_integrand_ax = plt.subplot(gs[0:2  ,  2:5])
        mid_integrand_ax = plt.subplot(gs[2:4  ,  2:5])
        low_integrand_ax = plt.subplot(gs[4:6  ,  2:5])
        upper_PT_ax = plt.subplot(gs[0:2   ,  5:7])
        lower_PT_ax = plt.subplot(gs[2:    ,  5:7])
        upper_PSA_ax = plt.subplot(gs[0:2   ,  7:9])
        lower_PSA_ax = plt.subplot(gs[2:    ,  7:9])
        
        # Plot all axis labels, adjust size and shape for ~ok~ saved formatting
        plt.suptitle("Float WMO\_ID: " + str(self.WMO_ID) + ", " +
                     "Latest day: "   + argo_profile.format_jd(self.julds[prof_indices[-1]]))
        SLA_ax.set_xlabel('Date')
        SLA_ax.set_ylabel("Relative Sea Level Anomaly (m)")
        low_integrand_ax.set_xlabel(steric_int_label)
        mid_integrand_ax.set_ylabel("Depth (m)")
        lower_PT_ax.set_xlabel("In-situ temperature (C)")
        lower_PT_ax.set_ylabel("Pressure (dbar)")
        lower_PSA_ax.set_xlabel("Absolute Salinity (g/kg)")
        lower_PSA_ax.set_ylabel("Pressure (dbar)")
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2, left=0.1, top=0.92)

        # MAP axis : Location over time
        map = self.aviso_map_plot(days[-1], map_ax)              # Plot the interpolated SLA values
        x, y = map(lons, lats)                                   # Get map float locations
        map.plot(x,y, label='Float location', color='black')     # Plot float locations

        # SLA_ax : Float Sea Level Anomaly vs. Time
        SLA_ax.plot(days, aviso_sla,  label='Aviso')             # Plot the aviso SLA
        SLA_ax.plot(days, steric_sla, label='Steric')            # Plot the steric SLA
        xticks = SLA_ax.get_xticks()                             # Get the automatic xticks
        form = [argo_profile.format_jd(xt) for xt in xticks]     # Convert into formatted dates
        SLA_ax.set_xticklabels(form, rotation=90)                # Use dates as tick labels
        SLA_ax.grid(True)                                        # Add grid
        SLA_ax.legend(loc='lower right')                         # Add legend

        # AXIS : Integrand axes
        # top_integrand_ax
        top_integrand_ax.axvline( .0002, color='crimson')
        top_integrand_ax.axvline(-.0002, color='crimson')
        for i in prof_indices:
            # Get the x and y values of the integrand and plot
            int_depth, integrand = self.steric_sla(i, return_integrand=True)  
            top_integrand_ax.plot(integrand, int_depth, color=colors[i])
        top_integrand_ax.invert_yaxis()
        top_integrand_ax.grid(True)
        top_integrand_ax.set_ylim([500, 0])
        plt.sca(top_integrand_ax)
        plt.xticks(rotation=20)
        # mid_integrand_ax
        mid_integrand_ax.axvline( .000025, color='crimson')
        mid_integrand_ax.axvline(-.000025, color='crimson')
        for i in prof_indices:
            # Get the x and y values of the integrand and plot
            int_depth, integrand = self.steric_sla(i, return_integrand=True)  
            mid_integrand_ax.plot(integrand, int_depth, color=colors[i])
        mid_integrand_ax.invert_yaxis()
        mid_integrand_ax.grid(True)
        mid_integrand_ax.set_ylim([2000,     500])
        mid_integrand_ax.set_xlim([ -.0002, .0002])
        plt.sca(mid_integrand_ax)
        plt.xticks(rotation=20)
        # low_integrand_ax
        for i in prof_indices:
            # Get the x and y values of the integrand and plot
            int_depth, integrand = self.steric_sla(i, return_integrand=True)  
            low_integrand_ax.plot(integrand, int_depth, color=colors[i])
        low_integrand_ax.invert_yaxis()
        low_integrand_ax.grid(True)
        low_integrand_ax.set_ylim([6000, 2000])
        low_integrand_ax.set_xlim([ -.000025, .000025])
        plt.sca(low_integrand_ax)
        plt.xticks(rotation=20)
        
        # AXIS: Pressure vs. Temperature
        # AXIS : upper_PT_ax : Pressure vs. Temperature for floats [low res]
        upper_PT_ax.axvline(1, color='crimson')
        upper_PT_ax.axvline(4, color='crimson')
        for i in prof_indices:
            upper_PT_ax.scatter(self.profiles[i].temperature,         # Plot the P/T profiles
                        self.profiles[i].pressure, s=1, color=colors[i])
        upper_PT_ax.invert_yaxis()                                    # Invert pressure axis to descending
        upper_PT_ax.grid(True)                                        # Add grid
        upper_PT_ax.set_ylim([2000, 0])                               # Zoom in on shallow part
        upper_PT_ax.yaxis.tick_right()
        # AXIS : lower_PT_ax : Pressure vs. Temperature for floats [high res]
        for i in prof_indices:
            lower_PT_ax.scatter(self.profiles[i].temperature,         # Plot the P/T profiles
                        self.profiles[i].pressure, s=1, color=colors[i])    
        lower_PT_ax.invert_yaxis()                                    # Invert pressure axis to descending
        lower_PT_ax.grid(True)                                        # Add grid
        lower_PT_ax.set_ylim([6000, 2000])                            # Zoom in on deep part
        lower_PT_ax.set_xlim([1, 4])                                  # Increase resolution
        lower_PT_ax.yaxis.tick_right()
        lower_PT_ax.yaxis.set_label_position("right")
        
        # AXIS: Pressure versus Absolute Salinity
        # AXIS : upper_PSA_ax : Pressure vs. absolute salinity for floats [low res]
        upper_PSA_ax.axvline(34.666, color='crimson')
        upper_PSA_ax.axvline(35.333, color='crimson')
        for i in prof_indices:
            upper_PSA_ax.scatter(self.profiles[i].SA,         # Plot the P/T profiles
                                self.profiles[i].pressure, s=1, color=colors[i])
        upper_PSA_ax.invert_yaxis()                                    # Invert pressure axis to descending
        upper_PSA_ax.grid(True)                                        # Add grid
        upper_PSA_ax.set_ylim([2000, 0])                               # Zoom in on shallow part
        upper_PSA_ax.yaxis.tick_right()
        # AXIS : lower_PSA_ax : Pressure vs. absolute salinity for floats [high res]
        for i in prof_indices:
            lower_PSA_ax.scatter(self.profiles[i].SA,         # Plot the P/T profiles
                                 self.profiles[i].pressure, s=1, color=colors[i])    
        lower_PSA_ax.invert_yaxis()                                    # Invert pressure axis to descending
        lower_PSA_ax.grid(True)                                        # Add grid
        lower_PSA_ax.set_ylim([6000, 2000])                            # Zoom in on deep part
        lower_PSA_ax.set_xlim([34.666, 35.333])                          # Increase resolution
        lower_PSA_ax.yaxis.tick_right()
        lower_PSA_ax.yaxis.set_label_position("right")
        
        # Save and close if desired
        if saveas != None:
            f.savefig(saveas)
            plt.close(f)
        else:
            plt.show()
    
    def plot_profiles_with_deltas(self, prof_indices='all', saveas=None):
        """
        Make a figure depicing information on a given profile, plus profiles
        coming before it.
        
        Plot the delta pt and SA instead of pressure and salinity
        
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
            
        # Get profiles information
        lats       = [self.profiles[i].latitude         for i in prof_indices]
        lons       = [self.profiles[i].longitude        for i in prof_indices]
        days       = [self.julds[i]                     for i in prof_indices]
        
        aviso_sla  = np.array([self.aviso_sla(i)  for i in prof_indices])
        steric_sla = np.array([self.steric_sla(i) for i in prof_indices])
        
        # Convert to relative sea level (to first measurement)
        aviso_sla = aviso_sla - aviso_sla[0]
        steric_sla = steric_sla - steric_sla[0]
        
        # Make formatted strings for axis labels
        rc('text', usetex=True)
        rc('font', family='serif')
        #steric_int_label = r"Steric Integrand ($\alpha \Delta T_{\textrm{potential}}}-\beta \Delta S_{{\textrm{absolute anomaly}}}$)"
        steric_int_label = 'Steric Integrand'
        
        # Get a color scheme so that the latest profiles plotted are darker.
        cmap = cm.get_cmap('binary')
        norm = matplotlib.colors.PowerNorm(vmin=1, vmax=5, gamma=3.0, clip=True)
        colors = [cmap(0.05+0.95*norm(i+1-len(prof_indices)+5)) for i in range(len(prof_indices))]
        
        # Get figure and axes, label with WMO_id and latest date
        f = plt.figure(figsize=(13,8))
        gs = gridspec.GridSpec(6, 9, figure=f, wspace=0.9, hspace=0.6)
        map_ax = plt.subplot(gs[ :-3   ,  0:2])
        SLA_ax = plt.subplot(gs[3:     ,  0:2])
        top_integrand_ax = plt.subplot(gs[0:2  ,  2:5])
        mid_integrand_ax = plt.subplot(gs[2:4  ,  2:5])
        low_integrand_ax = plt.subplot(gs[4:6  ,  2:5])
        upper_delpt_ax = plt.subplot(gs[0:2   ,  5:7])
        lower_delpt_ax = plt.subplot(gs[2:    ,  5:7])
        upper_delSA_ax = plt.subplot(gs[0:2   ,  7:9])
        lower_delSA_ax = plt.subplot(gs[2:    ,  7:9])
        
        # Plot all axis labels, adjust size and shape for ~ok~ saved formatting
        plt.suptitle("Float WMO\_ID: " + str(self.WMO_ID) + ", " +
                     "Latest day: "   + argo_profile.format_jd(self.julds[prof_indices[-1]]))
        SLA_ax.set_xlabel('Date'), 
        SLA_ax.set_ylabel("Relative Sea Level Anomaly (m)")
        low_integrand_ax.set_xlabel(steric_int_label)
        mid_integrand_ax.set_ylabel("Depth (m)")
        lower_delpt_ax.set_xlabel("Delta Potential Temperature (C)")
        lower_delpt_ax.set_ylabel("Depth (m)")
        lower_delSA_ax.set_xlabel("Delta Absolute Salinity (g/kg)")
        lower_delSA_ax.set_ylabel("Depth (m)")
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2, left=0.1, top=0.92)
        
        # MAP axis : Location over time
        map = self.aviso_map_plot(days[-1], map_ax)              # Plot the interpolated SLA values
        x, y = map(lons, lats)                                   # Get map float locations
        map.plot(x,y, label='Float location', color='black')     # Plot float locations
        
        # SLA_ax : Float Sea Level Anomaly vs. Time
        SLA_ax.plot(days, aviso_sla,  label='Aviso')             # Plot the aviso SLA
        SLA_ax.plot(days, steric_sla, label='Steric')            # Plot the steric SLA
        xticks = SLA_ax.get_xticks()                             # Get the automatic xticks
        form = [argo_profile.format_jd(xt) for xt in xticks]     # Convert into formatted dates
        SLA_ax.set_xticklabels(form, rotation=90)                # Use dates as tick labels
        SLA_ax.grid(True)                                        # Add grid
        SLA_ax.legend(loc='lower right')                         # Add legend
        
        # AXIS : Integrand axes
        # top_integrand_ax
        top_integrand_ax.axvline( .0002, color='crimson')
        top_integrand_ax.axvline(-.0002, color='crimson')
        for i in prof_indices:
            # Get the x and y values of the integrand and plot
            int_depth, integrand = self.steric_sla(i, return_integrand=True)  
            top_integrand_ax.plot(integrand, int_depth, color=colors[i])
        top_integrand_ax.invert_yaxis()
        top_integrand_ax.grid(True)
        top_integrand_ax.set_ylim([500, 0])
        plt.sca(top_integrand_ax)
        plt.xticks(rotation=20)
        # mid_integrand_ax
        mid_integrand_ax.axvline( .000025, color='crimson')
        mid_integrand_ax.axvline(-.000025, color='crimson')
        for i in prof_indices:
            # Get the x and y values of the integrand and plot
            int_depth, integrand = self.steric_sla(i, return_integrand=True)  
            mid_integrand_ax.plot(integrand, int_depth, color=colors[i])
        mid_integrand_ax.invert_yaxis()
        mid_integrand_ax.grid(True)
        mid_integrand_ax.set_ylim([2000,     500])
        mid_integrand_ax.set_xlim([ -.0002, .0002])
        plt.sca(mid_integrand_ax)
        plt.xticks(rotation=20)
        # low_integrand_ax
        for i in prof_indices:
            # Get the x and y values of the integrand and plot
            int_depth, integrand = self.steric_sla(i, return_integrand=True)  
            low_integrand_ax.plot(integrand, int_depth, color=colors[i])
        low_integrand_ax.invert_yaxis()
        low_integrand_ax.grid(True)
        low_integrand_ax.set_ylim([6000, 2000])
        low_integrand_ax.set_xlim([ -.000025, .000025])
        plt.sca(low_integrand_ax)
        plt.xticks(rotation=20)
        
        # AXIS: Pressure vs. Temperature
        # AXIS : upper_PT_ax : Pressure vs. Temperature for floats [low res]
        upper_delpt_ax.axvline(-0.2, color='crimson')
        upper_delpt_ax.axvline(0.2, color='crimson')
        for i in prof_indices:
            delta_pt, delta_SA = self.steric_sla(i, return_deltas=True) 
            upper_delpt_ax.plot(delta_pt, self.depth_avg, color=colors[i])
        upper_delpt_ax.invert_yaxis()                                    # Invert pressure axis to descending
        upper_delpt_ax.grid(True)                                        # Add grid
        upper_delpt_ax.set_ylim([2000, 0])                               # Zoom in on shallow part
        upper_delpt_ax.yaxis.tick_right()
        # AXIS : lower_PT_ax : Pressure vs. Temperature for floats [high res]
        for i in prof_indices:
            delta_pt, delta_SA = self.steric_sla(i, return_deltas=True) 
            lower_delpt_ax.plot(delta_pt, self.depth_avg, color=colors[i])    
        lower_delpt_ax.invert_yaxis()                                    # Invert pressure axis to descending
        lower_delpt_ax.grid(True)                                        # Add grid
        lower_delpt_ax.set_ylim([6000, 2000])                            # Zoom in on deep part
        lower_delpt_ax.set_xlim([-0.2, 0.2])                                  # Increase resolution
        lower_delpt_ax.yaxis.tick_right()
        lower_delpt_ax.yaxis.set_label_position("right")
        
        # AXIS: Pressure versus Absolute Salinity
        # AXIS : upper_PSA_ax : Pressure vs. absolute salinity for floats [low res]
        upper_delSA_ax.axvline(-0.02, color='crimson')
        upper_delSA_ax.axvline(0.02, color='crimson')
        for i in prof_indices:
            delta_pt, delta_SA = self.steric_sla(i, return_deltas=True) 
            upper_delSA_ax.plot(delta_SA, self.depth_avg, color=colors[i])
        upper_delSA_ax.invert_yaxis()                                    # Invert pressure axis to descending
        upper_delSA_ax.grid(True)                                        # Add grid
        upper_delSA_ax.set_ylim([2000, 0])                               # Zoom in on shallow part
        upper_delSA_ax.yaxis.tick_right()
        # AXIS : lower_PSA_ax : Pressure vs. absolute salinity for floats [high res]
        for i in prof_indices:
            delta_pt, delta_SA = self.steric_sla(i, return_deltas=True) 
            lower_delSA_ax.plot(delta_SA, self.depth_avg, color=colors[i])    
        lower_delSA_ax.invert_yaxis()                                    # Invert pressure axis to descending
        lower_delSA_ax.grid(True)                                        # Add grid
        lower_delSA_ax.set_ylim([6000, 2000])                            # Zoom in on deep part
        lower_delSA_ax.set_xlim([-0.02, 0.02])                        # Increase resolution
        lower_delSA_ax.yaxis.tick_right()
        lower_delSA_ax.yaxis.set_label_position("right")
        
        # Save and close if desired
        if saveas != None:
            f.savefig(saveas)
            plt.close(f)
        else:
            plt.show()
            
            

    def plot_prof_all(self,
                      savedir='/home/cassandra/docs/argo/movies/float/all_weekly/',
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
    
    def match_days(self):
        earliest_day = min(self.AI.days)
        latest_day   = max(self.AI.days)
        date_ok = np.logical_and(np.greater(self.julds, earliest_day),
                                 np.less(self.julds, latest_day))
        where_date_ok = np.where(date_ok)[0].tolist()
        return [self.julds[i] for i in where_date_ok]
        
    
    def plot_prof_relevant(self, deltas = False,
                           savedir='/home/cassandra/docs/argo/movies/float/rel_weekly/',
                           WMO_ID_dir=True):
        """
        Plot all the profiles which have aviso data overlap
        """
        # if deltas, modify savedir to be another directory
        if deltas: savedir = savedir[:-1]+'_deltas/'
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
            if deltas:
                self.plot_profiles_with_deltas(prof_indices = indices_to_use, 
                                               saveas       = (savedir+'f{:0>3d}.png'.format(i)))
            else:
                self.plot_profiles(prof_indices = indices_to_use, 
                                   saveas       = (savedir+'f{:0>3d}.png'.format(i)))




def main():
    """
    Make movies for every deep argo float in the northwest atlantic
    """    
    start_time = time.time()
    for WMO_ID in [4902321, 4902322, 4902323, 4902324, 4902325, 4902326]:
        
        print("Plotting float with WMO-ID " + str(WMO_ID))
        print("Time is " + str(int((time.time()-start_time)/60)) + " minutes.")
        sys.stdout.flush()
        
        #
        # Week interpolated Aviso data
        #
        af_weekly = ArgoFloat(WMO_ID, argo_dir="/data/deep_argo_data/nc/",
                              aviso_dir = '/data/aviso_data/nrt/weekly/')
        af_weekly.plot_prof_relevant(deltas=True)
        af_weekly.plot_prof_relevant()
        gc.collect()
        
        #
        # Month interpolated Aviso data
        #
        af_monthly = ArgoFloat(WMO_ID, argo_dir="/data/deep_argo_data/nc/",
                               aviso_dir = '/data/aviso_data/monthly_mean/')
        af_monthly.plot_prof_relevant(deltas=True, savedir='/home/cassandra/docs/argo/movies/float/rel_monthly/')
        af_monthly.plot_prof_relevant(savedir='/home/cassandra/docs/argo/movies/float/rel_monthly/')
        
        
if __name__ == '__main__':
    main()