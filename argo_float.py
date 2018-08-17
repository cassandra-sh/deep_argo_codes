#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:56:58 2018

@author: cassandra
"""

from mpl_toolkits.basemap  import Basemap
from matplotlib            import rc
from matplotlib.pyplot     import cm
from matplotlib.mlab       import griddata
from matplotlib.offsetbox  import AnchoredText
from scipy.fftpack         import fft
from decimal               import Decimal
import matplotlib.patheffects as PathEffects
import matplotlib.gridspec    as gridspec
import matplotlib.colors
import matplotlib
import scipy.interpolate
import matplotlib.pyplot   as plt
import numpy               as np
import pandas              as pd
import aviso_interp
import argo_profile
import glob
import time
import sys
import gc
import os
from scipy import signal

def one_sided_psd(x, y, x_is_days=True):
    """
    Compute the Power Spectral Density
    
    x must be an evenly spaced time sequence in days. 
    
    returns x, y in period space
    """
    # Get some sampling information
    nsample = len(x)
    dt = abs(x[-1] - x[-2])
    T = nsample * dt
    
    # Detrend
    dty = signal.detrend(y)
    
    # Compute the fast fourier transform
    yf = fft(dty)
    
    # Transfer the fft to the one sided periodigram
    yf = 2/T*abs(yf[0:int(nsample/2)])**2
    
    # Get the grid spacing to return
    xf = np.linspace(0.0, 1.0/(2.0*dt), nsample/2) 
    
    # Turn the grid into period space
    xf = xf ** (-1)
    
    return xf, yf

def grid(x, y, z, resX=100, resY=100):
    "Convert 3 column data to matplotlib grid"
    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi)
    X, Y = np.meshgrid(xi, yi)
    return X, Y, Z

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
        self.pres_range = np.arange(0, 6001.0, 1.0)
        psal_interps = np.array([prof.interp_psal(self.pres_range) for prof in self.profiles])
        temp_interps = np.array([prof.interp_temp(self.pres_range) for prof in self.profiles])
        psal_avg = np.nanmean(psal_interps, axis=0)  # take the average of all interpolated
        temp_avg = np.nanmean(temp_interps, axis=0)  # values which are NOT nan (the fill value)
                                                     # i.e. where there IS data
        # Make the average profile object
        self.prof_avg = argo_profile.Profile(temperature = temp_avg,
                                             pressure    = self.pres_range,
                                             psal        = psal_avg,
                                             longitude   = self.lon_avg,
                                             latitude    = self.lat_avg)
        self.depth_avg = -1.0*self.prof_avg.get_z()
    
        # STEP 3. GET INTERPOLATORS
        # Get an aviso interpolator within a degree range of this location
        self.AIbox = [self.lon_avg-10.0, self.lon_avg+10.0,
                      self.lat_avg-10.0, self.lat_avg+10.0]
        aviso_dir = kwargs.get('aviso_dir', '/data/aviso_data/nrt/weekly/')
        self.AI = aviso_interp.AvisoInterpolator(box=self.AIbox, irregular=False,
                                                 limit=100, verbose=False,
                                                 aviso_dir = aviso_dir,
                                                 units = 'm')
        # Make time profile interpolators
        temps = np.array([prof.interp_temp(self.pres_range) for prof in self.profiles])
        psals = np.array([prof.interp_psal(self.pres_range) for prof in self.profiles])
        self.temp_interp = scipy.interpolate.RegularGridInterpolator((self.julds, self.pres_range), 
                                                                     temps, 
                                                                     fill_value=np.nan, 
                                                                     bounds_error=False)
        self.psal_interp = scipy.interpolate.RegularGridInterpolator((self.julds, self.pres_range), 
                                                                     psals, 
                                                                     fill_value=np.nan, 
                                                                     bounds_error=False)
        self.lon_interp = scipy.interpolate.interp1d(self.julds, self.lons, fill_value=np.nan, bounds_error=False)
        self.lat_interp = scipy.interpolate.interp1d(self.julds, self.lats, fill_value=np.nan, bounds_error=False)
    
    def profs_to_date(self, day):
        """
        Get the profile indices for every profile up to the given day (in juld)
        """
        return np.where(self.julds < day)[0]
    
    def iso_compute(self, iso_param, iso_value, other_param):
        """
        Given some parameter (iso_param) to be held constant at some value
        (iso_value), find the values for some other parameter (other_param) 
        over time. 
        
        Given multiple matches, take the values closest to the previous value.
        If there are multiple first matches, take the UPPERMOST match
        Might add more algorithms for determining the best values to take later
        
        np.nan is used if no valid matches are available for the given day. 
        
        @returns 
            vals for other_param for self.julds determined by iso_compute
        """
        
        mlists = [prof.find_matches(iso_param, iso_value, other_param) for prof in self.profiles]
        chosen, latest = [], None
        
        for mlist in mlists:
            if len(mlist) == 0:                        # Case of no matches.
                chosen.append(np.nan)                  # Take np.nan.
            elif len(mlist) == 1:                      # Case of only one match.
                chosen.append(mlist[0])                # Take the match.
                latest = mlist[0]                      # At any point when taking a non
            else:                                      # np.nan value, update latest.
                if latest == None:                     # Case of multiple matches but no 
                    chosen.append(max(mlist))          # previous value to compare to.
                    latest = max(mlist)                # Take the max of the matches.
                else:                                  # Case of multiple matches and
                    delta = abs(mlist[0] - latest)     # a previous value.
                    best = mlist[0]                    # Take value closest to previous
                    for m in mlist:                    # value taken that wasn't np.nan. 
                        if delta > abs(m - latest):
                            delta = abs(m - latest)
                            best = m
                    chosen.append(best)
                    latest = best
        return chosen
        
    def iso_waterfall(self, iso_param, iso_values, other_param, n=None, date_max=None, date_min=None): 
        """
        Given some lines of constant values (iso_values) for some profile 
        parameter (iso_param), get the values of some other parameter as they
        vary over time (e.g. the depth of isotherms), and compute the power
        spectrum, preparing the data for an imshow waterfall diagram.
        
        @returns x, y, x
            x - period (days)
            y - iso_values given, the y axis values
            z - power of other_param (units = squared (other_param unit) times days)
        """
        if n == None:
            n = 10*len(self.julds)
        
        # 1. Generate the list of values for each day
        ovals_list = [self.iso_compute(iso_param, iso_value, other_param) for iso_value in iso_values]
        
        # 2. Figure out min and max days
        min_day = min(self.julds)
        max_day = max(self.julds)
        if date_max is not None:  max_day = date_max
        if date_min is not None:  min_day = date_min
        
        # 3. Drop nan values
        julds_list = []
        for i in range(len(ovals_list)):
            julds_list.append(np.array(self.julds)[~np.isnan(ovals_list[i])])
            ovals_list[i] = np.array(ovals_list[i])[~np.isnan(ovals_list[i])]
        
        # 4. If any iso values are fully nan, drop them
        good = [len(ovals) != 0 for ovals in ovals_list]
        ovals_list = np.array(ovals_list)[good]
        julds_list = np.array(julds_list)[good]
        iso_values = np.array(iso_values)[good]
        
        # 5. Interpolate the values onto an even day spaced grid
        days_grid = np.linspace(min_day, max_day, n)
        oval_grids = [np.interp(days_grid, julds, ovals) for ovals, julds in zip(ovals_list, julds_list)]
        
        # 6. Generate the FFT periodigram for each iso line
        periods, power_vals = None, []
        for oval_grid in oval_grids:
            periods, powers = one_sided_psd(days_grid, oval_grid)
            power_vals.append(powers)
        
        # 7. Return
        return  periods, iso_values, np.array(power_vals).T
        
    def match_days(self):
        """
        
        """
        earliest_day = min(self.AI.days)
        latest_day   = max(self.AI.days)
        date_ok = np.logical_and(np.greater(self.julds, earliest_day),
                                 np.less(self.julds, latest_day))
        where_date_ok = np.where(date_ok)[0].tolist()
        return [self.julds[i] for i in where_date_ok]
    
    def day_profile(self, day):
        """
        Interpolate from the set of argo profiles a new profile for a given day
        on the track of available argo profiles. 
        """
        return argo_profile.Profile(temperature = self.temp_interp((day, self.pres_range)),
                                    pressure    = self.pres_range,
                                    psal        = self.psal_interp((day, self.pres_range)),
                                    longitude   = self.lon_interp(day),
                                    latitude    = self.lat_interp(day))

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
        delta_pt = self.profiles[prof_index].pt_interp(self.pres_range) - self.prof_avg.pt
        delta_SA = self.profiles[prof_index].SA_interp(self.pres_range) - self.prof_avg.SA
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
        
    
    """
    SOFTCODED PLOT ROUTINES
    """
    
    def steric_by_depth(self, axis, prof_indices, depth_bins=[6000, 2000, 0]):
        """
        
        """
        # Get the basic steric sla data for all depth
        dates = np.array([self.julds[i] for i in prof_indices])
        aviso_sla  = np.array([self.aviso_sla(i)  for i in prof_indices])
        aviso_sla = aviso_sla - np.nanmean(aviso_sla) #aviso_sla[0] - 0.05
        full_sterics = np.array([self.steric_sla(i)  for i in prof_indices])
        full_sterics = full_sterics - np.nanmean(full_sterics)#full_sterics[0]
        # Get colors for each bin
        depth_colors = []
        cmap =  cm.get_cmap('PuBuGn')
        for j in range(1, len(depth_bins)):
            depth_colors.append(cmap((j-1)/(len(depth_bins)-1)))
        # Get the integrands (and depths) for each profile measured
        integrands, depths = [], []
        for i in prof_indices:
            d, s = self.steric_sla(i, return_integrand=True)
            integrands.append(s)
            depths.append(d)
        depths, integrands = np.array(depths), np.array(integrands)
        # Iterate through each bin
        for j in range(1, len(depth_bins)):
            # Get min and max bin values
            depth_bin = [depth_bins[j-1], depth_bins[j]]
            bin_min = min(depth_bin)
            bin_max = max(depth_bin)
            # Prep a holder for the depth's sla
            bin_sla = []
            # Iterate through each profile's steric integrand
            for i in range(len(dates)):
                # Make a mask that gets only the allowed depths
                mask = np.logical_and(np.greater(depths[i], bin_min),
                                      np.less(depths[i], bin_max))
                # Mask the integrand and depth for this profile
                integrand, int_depth = integrands[i][mask], depths[i][mask]
                # And integrate, adding to the holder
                bin_sla.append(np.trapz(integrand, x=int_depth))        
            # Cast as numpy array
            sla = np.array(bin_sla)
            # Plot, filling difference between 0 and sla
            # Make sure to adjust the sla to relative to first measurement
            axis.fill_between(dates, 0, sla - np.nanmean(sla),#[0], 
                              color=depth_colors[j-1], zorder=len(depth_bins)-j,
                              label=('z between '+str(bin_min)+' and '+str(bin_max)))             
        # Plot aviso sla on top, adding a legend
        axis.plot(dates, aviso_sla, color='orange', label='Aviso sla')
        axis.plot(dates, full_sterics, color='green', label='Full steric')
        axis.legend()
    
    def aviso_map_plot(self, date, axis, extracolors = 'black'):
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
        map.drawparallels(parallels, labels=[True, False, False, False], color=extracolors, textcolor=extracolors)
        meridians = np.arange(10.,351.,20.)
        map.drawmeridians(meridians, labels=[False, False, False, True], color=extracolors, textcolor=extracolors)
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
    
    def axplot_avgspectrum(self, axis, iso_param, iso_values, other_param, **kwargs):
        """
        
        """
        ax_xlims      = kwargs.get('ax_xlims'     ,   None)
        ax_ylims      = kwargs.get('ax_ylims'     ,   None)
        ax_xlabel     = kwargs.get('ax_xlabel'    ,   True)
        ax_ylabel     = kwargs.get('ax_ylabel'    ,   True)
        
        plt.sca(axis)
        if ax_xlabel: axis.set_xlabel("Frequency (1/days)")
        if ax_ylabel: axis.set_ylabel(argo_profile.attr_to_name(other_param) + " squared times days")
        
        xvals, yvals, zvals = self.iso_waterfall(iso_param, iso_values, other_param)
        
        #Transfer xvals back to frequency
        xvals = 1.0/xvals
        
        lab = ('Average Periodigram for ' + argo_profile.attr_to_name(other_param) +
               ' on lines of equal ' + argo_profile.attr_to_name(iso_param) + " for " +
               argo_profile.attr_to_symb(iso_param) + r" $\in$ (" + str(min(iso_values)) +
               ', ' + str(max(iso_values)) + ")")
        
        power_values = np.nanmean(zvals, axis=1)
        power_errs = np.nanstd(zvals, axis=1)
        axis.errorbar(xvals, power_values, yerr=power_errs, color='magenta',zorder=1)#, label=lab)
        axis.plot(xvals, power_values, color='white', label=lab, zorder=2)
        axis.set_xscale("log", nonposx='clip')
        axis.set_yscale("log", nonposy='clip')
        axis.legend()
               
        if ax_xlims is not None: axis.set_xlim(ax_xlims[0], ax_xlims[1])
        else:                    axis.set_xlim(np.nanmin(xvals[xvals != -np.inf]),
                                               np.nanmax(xvals[xvals != np.inf]))  
        if ax_ylims is not None: axis.set_ylim(ax_ylims[0], ax_ylims[1])
        else:                    axis.set_ylim(np.nanmin(power_values[power_values != -np.inf]),
                                               np.nanmax(power_values[power_values != np.inf]))
        
    def axplot_spectrogram(self, axis, iso_param, iso_values,
                           other_param, **kwargs):
        """
        
        
        @params
            axis          - (matplotlib axis) Plot the iso lines onto this axis
            iso_param     - (str) This is the parameter name that designates
                            what parameter is held constant
            iso_values    - (list of floats) These are the values of said 
                            parameter at which it will be held constant
            other_param   - (str) This is the parameter name that will be
                            computed for lines of constant [iso_param]
        
        **kwargs:
            cbar        = True   - Whether or not to plot the colorbar 
            ax_xlabel   = True   - Whether or not to display the x label (period for power spectrum)
            ax_ylabel   = True   - Whether or not to display the y label (iso_param name)
            vmin        = None   - colorbar vmin 
            vmax        = None   - and vmax
            cmap_name   = 'jet'  - name of matplotlib colormap to use
            norm      [unimplemented yet]
        
        @returns
            whatever matplotlib.pyplot.contourf returns                
        """
        ax_xlims      = kwargs.get('ax_xlims'     ,   None)
        ax_ylims      = kwargs.get('ax_ylims'     ,   None)
        cbar          = kwargs.get('cbar'         ,   True)
        ax_xlabel     = kwargs.get('ax_xlabel'    ,   True)
        ax_ylabel     = kwargs.get('ax_ylabel'    ,   True)
        vmin          = kwargs.get('vmin'         ,   None)
        vmax          = kwargs.get('vmax'         ,   None)
        cmap_name     = kwargs.get('cmap_name'    ,  'jet')
        
        plt.sca(axis)
        if ax_xlabel: axis.set_xlabel("Period (days)")
        if ax_ylabel: axis.set_ylabel(argo_profile.attr_to_name(iso_param))
        
        xvals, yvals, zvals = self.iso_waterfall(iso_param, iso_values, other_param)
        ct = axis.contourf(xvals, yvals, zvals.T, np.logspace(-1, 10, 500), 
                           cmap = cmap_name, vmin=vmin, vmax=vmax, 
                           norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
        
        if cbar:
            cbar = plt.colorbar(ct, ticks = np.logspace(-1, 10, 12))
            cbar.set_label(argo_profile.attr_to_name(iso_param) +
                           " squared times days (periodigram)")
            ftick = ['%.2E' % Decimal(flt) for flt in np.logspace(-1, 10, 12)]
            cbar.ax.set_yticklabels(ftick)
        
        if ax_xlims is not None: axis.set_xlim(ax_xlims[0], ax_xlims[1])
        else:                    axis.set_xlim(np.nanmin(xvals[xvals != -np.inf]), np.nanmax(xvals[xvals != np.inf]))  
        if ax_ylims is not None: axis.set_ylim(ax_ylims[0], ax_ylims[1])
        else:                    axis.set_ylim(np.nanmin(yvals[yvals != -np.inf]), np.nanmax(yvals[yvals != np.inf]))
        return ct
    
    def axplot_isos(self, axis, prof_indices, iso_param, iso_values,
                    other_param, **kwargs):
        """
        Given some param (iso_param), and values (iso_values), compute the
        values of another parameter (other_param) on lines of constant 
        iso_param. Do this for the profiles whose indices are given in 
        prof_indices. 
        
        Then [if remove_mean == True], subtract the mean value from each 
        iso line, stacking them on top of each other. 
        
        Color the lines by the value of iso_param being held constant for each
        line if [val_color == True]
        
        @params
            axis          - (matplotlib axis) Plot the iso lines onto this axis
            prof_indices  - (list of ints) Plot iso line values for profiles 
                            corresponding to these indices
            iso_param     - (str) This is the parameter name that designates
                            what parameter is held constant
            iso_values    - (list of floats) These are the values of said 
                            parameter at which it will be held constant
            other_param   - (str) This is the parameter name that will be
                            computed for lines of constant [iso_param]
        
        **kwargs:
            val_color   = True   - Whether or not to color lines based on 
                                   iso_param value
            remove_mean = True   - Whether or not to subtract mean from iso 
                                   lines and plot on top of each other
            ax_xlims    = None   - x lims for axis (None -> infer) otherwise
                                   list or tuple of length 2 of [min, max]
            ax_ylims    = None   - y lims for axis (None -> infer) otherwise
                                   list or tuple of length 2 of [min, max]
            cbar        = True   - Whether or not to plot the colorbar 
                                   (iso_param values) val_color must be True 
                                   as well
            ax_xlabel   = True   - Whether or not to display the x label (time)
            ax_ylabel   = True   - Whether or not to display the y label 
                                   (other_param)
            ax_xticklabel = True - Whether or not to plot the (large) date 
                                   ticklabels on the x axis
            vmin        = None   - colorbar vmin 
            vmax        = None   - and vmax
            cmap_name   = 'jet'  - name of matplotlib colormap to use
        
        @returns
            if val_color = False, return None
            otherwise, return the ScalarMappable associated with this plot. 
            
            Note that the ScalarMappable is generated by normalizing the 
            iso_values to a 0 to 1 range. if vmin and vmax are not given, 
            will just use the min and max of iso_values. Give vmin and vmax if
            you are intending to make a colorbar with multiple axes and are
            plugging in different iso_values for multiple calls of this method.         
                
        """
        val_color     = kwargs.get('val_color'    ,   True)
        remove_mean   = kwargs.get('remove_mean'  ,   True)
        ax_xlims      = kwargs.get('ax_xlims'     ,   None)
        ax_ylims      = kwargs.get('ax_ylims'     ,   None)
        cbar          = kwargs.get('cbar'         ,   True)
        ax_xlabel     = kwargs.get('ax_xlabel'    ,   True)
        ax_ylabel     = kwargs.get('ax_ylabel'    ,   True)
        ax_xticklabel = kwargs.get('ax_xticklabel',   True)
        vmin          = kwargs.get('vmin'         ,   None)
        vmax          = kwargs.get('vmax'         ,   None)
        cmap_name     = kwargs.get('cmap_name'    ,  'jet')
        
        plt.sca(axis)
        if ax_ylabel: axis.set_ylabel(argo_profile.attr_to_name(other_param))
        if ax_xlabel: axis.set_xlabel("Time")
        
        xvals = np.array([self.julds[i] for i in prof_indices])
        yvals = np.array([self.iso_compute(iso_param, iso_value, other_param) for iso_value in iso_values])
        sm = None
        
        if remove_mean: yvals = (yvals.T - np.nanmean(yvals, axis=1)).T
        
        if val_color:
            min_iso_val = min(iso_values)
            max_iso_val = max(iso_values)
            if vmin is not None: min_iso_val = vmin
            if vmax is not None: max_iso_val = vmax
            del_iso_values = max_iso_val - min_iso_val
            
            cvals = (np.array(iso_values)-min_iso_val)/del_iso_values
            
            cmap = cm.get_cmap(cmap_name)
            for i in range(len(yvals)):
                plt.plot(xvals, yvals[i], color=cmap(cvals[i]))
            
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm._A = []
            if cbar:
                if len(iso_values) < 10:
                    cbar = plt.colorbar(sm, ticks=cvals)
                    cbar.ax.set_yticklabels(iso_values)
                else:
                    cbar = plt.colorbar(sm, ticks=np.linspace(0, 1, num=10))
                    cbar.ax.set_yticklabels(np.linspace(min_iso_val, max_iso_val, num=10))
                cbar.set_label(argo_profile.attr_to_name(iso_param), rotation=270, labelpad=15)
                for cv in cvals:
                    cbar.ax.axhline(cv, color='white', zorder=4, alpha=0.5)
            
        else:
            for i in range(len(yvals)):
                plt.plot(xvals, yvals[i])
        
        xticks = axis.get_xticks()
        if ax_xticklabel: axis.set_xticklabels([argo_profile.format_jd(xt) for xt in xticks], rotation=90)
        else:             axis.set_xticklabels(['' for tick in xticks])
        if ax_xlims is not None: axis.set_xlim(ax_xlims[0], ax_xlims[1])
        else:                    axis.set_xlim(np.nanmin(xvals), np.nanmax(xvals))  
        if ax_ylims is not None: axis.set_ylim(ax_ylims[0], ax_ylims[1])
        else:                    axis.set_ylim(np.nanmin(yvals), np.nanmax(yvals))
        return sm
        
    def axplot_depthval(self, axis, prof_indices, depth, param, **kwargs):
        """
        Plot the values of a given Profile param at given depth level over time
        onto the given axis
        """
        ax_xlims      = kwargs.get('ax_xlims'     ,   None)
        ax_ylims      = kwargs.get('ax_ylims'     ,   None)
        delta         = kwargs.get('delta'        ,  False)
        ax_xlabel     = kwargs.get('ax_xlabel'    ,   True)
        ax_ylabel     = kwargs.get('ax_ylabel'    ,   True)
        ax_xticklabel = kwargs.get('ax_xticklabel',   True)
        depth_label   = kwargs.get('depth_label'  ,   True)
        
        plt.sca(axis)
        if ax_ylabel:
            if delta:
                axis.set_ylabel("Delta " + argo_profile.attr_to_name(param))
            else:
                axis.set_ylabel(argo_profile.attr_to_name(param))
        if ax_xlabel:
            axis.set_xlabel("Time")
        
        xvals = np.array([self.julds[i] for i in prof_indices])
        pressures = [self.profiles[i].depth_to_pressure(depth) for i in prof_indices] 
        yvals = np.array([getattr(self.profiles[i], param+'_interp')(pressures[i]) for i in prof_indices])
        if delta:
            yvals = yvals - getattr(self.prof_avg, param+'_interp')(self.prof_avg.depth_to_pressure(depth))
        
        axis.plot(xvals, yvals)
        axis.grid(True)
            
        if depth_label:
            anchored_text = AnchoredText("Depth = "+str(depth)+" m", loc=2, prop={'color':'black'})
            axis.add_artist(anchored_text)

        xticks = axis.get_xticks()
        if ax_xticklabel: axis.set_xticklabels([argo_profile.format_jd(xt) for xt in xticks], rotation=90)
        else:             axis.set_xticklabels(['' for tick in xticks])
        if ax_xlims is not None: axis.set_xlim(ax_xlims[0], ax_xlims[1])
        else:                    axis.set_xlim(np.nanmin(xvals), np.nanmax(xvals))  
        if ax_ylims is not None: axis.set_ylim(ax_ylims[0], ax_ylims[1])
        else:                    axis.set_ylim(np.nanmin(yvals), np.nanmax(yvals))
    
    def axplot_iso_diffs(self, axis, prof_indices, param, pvals, **kwargs):
        """
        
        """
        # kwargs get
        ax_xlims      = kwargs.get('ax_xlims'      ,     None)
        ax_ylims      = kwargs.get('ax_ylims'      ,     None)
        ax_xlabel     = kwargs.get('ax_xlabel'     ,     True)
        ax_ylabel     = kwargs.get('ax_ylabel'     ,     True)
        ax_xticklabel = kwargs.get('ax_xticklabel' ,     True)
        cbar          = kwargs.get('cbar'          ,     True)
        cval          = kwargs.get('cval'          ,  'param')
        
            
        # Get time values (xvals) for x axis
        xvals = np.array([self.julds[i] for i in prof_indices])
        
        # Interpolate the depth for each param level for each profile
        depths = [[self.profiles[i].depth_param(param, val) for i in prof_indices] for val in pvals]
        
        # And take the difference as the y values
        yvals = np.diff(depths, axis=0)
        
        # Then adapt the pvals to the midpoints between the existing ones
        pvals = (pvals[1:] + pvals[:-1]) / 2
        
        # Establish a color value
        clabel = ''
        if cval == 'depth_avg':
            cvals = [self.prof_avg.depth_param(param, pval) for pval in pvals]
            clabel = 'Depth (m)'
        elif cval == 'param':
            cvals = pvals
            clabel = argo_profile.attr_to_name(param)
        cvals = np.array(cvals)
        
        # Plot the data
        del_cval = max(cvals) - min(cvals)
        cmap = cm.get_cmap(kwargs.get('cmap', 'jet'))
        for i in range(len(yvals)):
            plt.plot(xvals, yvals[i], color=cmap( (cvals[i]-min(cvals)) / del_cval))
        
        # Colorbar/color mapping stuff
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm._A = []
        if cbar:
            cbar = plt.colorbar(sm, ticks=np.linspace(0, 1, num=10))
            val_range = np.linspace(min(cvals), max(cvals), num=10)
            cbar.ax.set_yticklabels(val_range)
            cbar.set_label(clabel, rotation=270, labelpad=15)
            for cv in cvals:
                cbar.ax.axhline( ((cv - min(cvals)) / del_cval),
                                color='white', zorder=4, alpha=0.5)
                
        # Axis/Labels/Ticks/Limits formatting
        axis.grid(True)
        if ax_ylabel: axis.set_ylabel("Depth Difference (m)")
        if ax_xlabel: axis.set_xlabel("Time")
        xticks = axis.get_xticks()
        if ax_xticklabel: axis.set_xticklabels([argo_profile.format_jd(xt) for xt in xticks], rotation=90)
        else:             axis.set_xticklabels(['' for tick in xticks])
        if ax_xlims is not None: axis.set_xlim(ax_xlims[0], ax_xlims[1])
        else:                    axis.set_xlim(np.nanmin(xvals), np.nanmax(xvals))  
        if ax_ylims is not None: axis.set_ylim(ax_ylims[0], ax_ylims[1])
        else:                    axis.set_ylim(np.nanmin(yvals), np.nanmax(yvals))
        
        return sm
            
    def axplot_prof(self, axis, prof_indices, type, x_param, y_param, **kwargs):
        """
        @params
            axis    - matplotlib axis to plot on
            prof_indices - profile indices to plot up to
            
            type    - option for what z axis (coloring) will occur
                        'time'      - Latest profile in black, older = greyer
                        'seasonal'  - Each month has a unique color on a scale
                        'isotherms' - z axis (color) corresponds to temperature
                                      requires x_param be 'time' and y_param be 
                                      'depth' or 'pressure'
                        
            x_param - Profile param name or 'time' for x values. 
                      If x_param = 'time', type must = 'isotherms' 
            y_param - Profile param name for y values
            
        **kwargs
            hlines  = None  - horizontal lines to plot
            vlines  = None  - vertical lines to plot
            vmin    = None  - colorbar vmin 
            vmax    = None  - and vmax
            ax_xlims  = None  - x lims for axis
            ax_ylims  = None  - y lims for axis
            
            deltay  = False - Whether or not to plot differential from the mean 
                             value of this parameter instead of absolute value
            deltax  = False - as deltay
            
            isotype = 'potential', 'delta potential', 'conservative', or 'absolute'
                      What kind of temperature will be plotted as isotherms
                      if type = 'isotherm' - default is 'potential'
                      
            cbar    = True  - Whether or not to plot the colorbar
            axlabel = True  - Whether or not to label the axes
            cmap    = varies - to use instead of default
        
        @returns
            if type = 'isotherm', returns what contourf returns
            if type = 'seasonal', returns the seasonal colorbar
        """
        
        ax_xlims  = kwargs.get('ax_xlims',  None)
        ax_ylims  = kwargs.get('ax_ylims',  None)
        
        hlines  = kwargs.get('hlines',  None)
        vlines  = kwargs.get('vlines',  None)
        deltax  = kwargs.get('deltax',  False)
        deltay  = kwargs.get('deltay',  False)
        cbar    = kwargs.get('cbar',    True)
        axlabel = kwargs.get('axlabel', True)
        
        contour_levels = kwargs.get('contour_levels', np.arange(1.5, 1.901, .02))
        
        to_ret = None
        
        # Set axis labels if desired
        plt.sca(axis)
        if axlabel:
            if x_param == 'time':
                axis.set_xlabel('Date')
            else:
                axis.set_xlabel(argo_profile.attr_to_name(x_param))
            axis.set_ylabel(argo_profile.attr_to_name(y_param))
        
        # Get x and y values
        pres = self.pres_range
        xvals, yvals = [], []
        if x_param == 'time':
            xvals = np.array([self.julds[i] for i in prof_indices])
        else:
            xvals = np.array([getattr(self.profiles[i], x_param+'_interp')(pres) for i in prof_indices])
        yvals = np.array([getattr(self.profiles[i], y_param+'_interp')(pres) for i in prof_indices])
        
        # If requested, adjust to relative [to mean] values
        if deltax: 
            xvals = xvals - getattr(self.prof_avg, x_param+'_interp')(pres)
        if deltay: 
            yvals = yvals - getattr(self.prof_avg, y_param+'_interp')(pres)
        
        # Case where iso lines of some parameter is the z axis, and we are plotting contours
        if type == 'isotherms':
            
            # Figure out what kind of temperature will form the isotherm
            isotype = kwargs.get('isotype', 'potential')
            zvals = []
            if isotype == 'potential':
                zvals = np.array([self.profiles[i].pt_interp(pres) for i in prof_indices])
            elif isotype == 'absolute':
                zvals = np.array([self.profiles[i].temperature_interp(pres) for i in prof_indices])
            elif isotype == 'conservative':
                zvals = np.array([self.profiles[i].CT_interp(pres) for i in prof_indices])
            elif isotype == 'delta potential':
                zvals = np.array([self.profiles[i].pt_interp(pres) for i in prof_indices])
                zvals = zvals - self.prof_avg.pt_interp(pres)
                
            # Contour plot
            ct = axis.contourf(xvals, np.nanmean(yvals, axis=0), zvals.T, 150,
                               cmap=kwargs.get('cmap', 'jet'), 
                               extent=[np.nanmin(xvals), np.nanmax(xvals),
                                       np.nanmin(yvals), np.nanmax(yvals)],
                               vmin=kwargs.get('vmin', None),
                               vmax=kwargs.get('vmax', None))
            cs = axis.contour(xvals, np.nanmean(yvals, axis=0), zvals.T,
                              levels=contour_levels,
                              colors='white', 
                              extent=[np.nanmin(xvals), np.nanmax(xvals),
                                      np.nanmin(yvals), np.nanmax(yvals)],
                              vmin=kwargs.get('vmin', None),
                              vmax=kwargs.get('vmax', None))
            clabels = plt.clabel(cs, inline=True, fontsize=10, color='white')
            
            for l in clabels:
                l.set_rotation(0)
                l.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='k')])
            to_ret = (ct, cs)
            
            # If no axis lims given, adjust axis to center on data
            if ax_xlims is None:
                axis.set_xlim(np.nanmin(xvals), np.nanmax(xvals))    
            if ax_ylims is None:
                axis.set_ylim(np.nanmin(yvals), np.nanmax(yvals))
            
            # Add colorbar if preferred
            if cbar:
                cbar = plt.colorbar(ct)
                cbar.set_label(isotype+" temperature (C)", rotation=270, labelpad=15)
                cbar.add_lines(cs)
        
        # Case where some kind of time is the z axis, and we are just color 
        # coding the profiles by time of year or from latest to earliest
        elif type in ['seasonal', 'time', 'time_all']:
            
            # Start by prepping a color list for the profiles, and a date list if necessary
            colors   = []
            day_list = []
            
            # In the case of type = 'time', make profiles greyer the further back in time
            if type == 'time': 
                cmap = cm.get_cmap(kwargs.get('cmap', 'binary'))
                norm = matplotlib.colors.PowerNorm(vmin=1, vmax=5, gamma=3.0, clip=True)
                colors = [cmap(0.05+0.95*norm(i+1-len(prof_indices)+5)) for i in range(len(prof_indices))]
            
            # In the case of 'time_all', use a 'jet' color scheme from earliest time to latest time
            if type == 'time_all':
                cmap = cm.get_cmap(kwargs.get('cmap', 'jet'))
                colors = [cmap(i/(len(prof_indices)+1)) for i in range(len(prof_indices))]
                day_list = [self.profiles[i].julian_day for i in prof_indices]
            
            # Otherwise, make profiles a color based on the time of year
            elif type == 'seasonal':
                # Get datetimes from the julian days for the relevant profiles
                datetimes = [pd.to_datetime(self.profiles[i].datetime) for i in prof_indices]
                # Adjust to number of days since July 1 of each year
                days_since_newyear = np.array([datetime.dayofyear for datetime in datetimes])
                days_since_july    = days_since_newyear - 182
                days_since_july[days_since_july<0] = days_since_july[days_since_july<0]+365
                # Adjust to 0 to 1 scale
                date_scale = (days_since_july / 365.0)# + 0.5
                #date_scale[date_scale > 1.0] = date_scale[date_scale > 1.0] - 1.0
                # Map onto colormap
                cmap = cm.get_cmap(kwargs.get('cmap', 'hsv_r'))
                colors = [cmap(ds) for ds in date_scale]
                
            # Now scatter plot the profiles using that color scheme
            for i in prof_indices:
                axis.plot(xvals[i], yvals[i], color=colors[i])
            
            # And make a colorbar if seasonal is used and colorbar is requested
            if type == 'seasonal' or type == 'time_all':
                sm = plt.cm.ScalarMappable(cmap=cmap)
                sm._A = []
                to_ret = sm
                
                if cbar:
                    if type == 'seasonal':
                        cbar = plt.colorbar(sm, ticks=np.linspace(0, 1, num=13))
                        cbar.ax.set_yticklabels(['July', 'August', 'September',
                                                 'October',  'November', 'December',
                                                 'January', 'February', 'March',
                                                 'April', 'May', 'June', 'July'])
                        cbar.set_label('Month', rotation=270)
                    elif type == 'time_all':
                        cbar = plt.colorbar(sm, ticks=np.linspace(0, 1, num=10))
                        day_range = np.linspace(min(day_list), max(day_list), num=10)
                        cbar.ax.set_yticklabels([argo_profile.format_jd(day) for day in day_range])
                
                
                
    
        # If horizontal lines are given, plot them
        if hlines is not None:
            for hline in hlines:
                axis.axhline(hline, color='crimson')
                
        # If vertical lines are given, plot them
        if vlines is not None:
            for hline in hlines:
                axis.axvline(hline, color='crimson')
        
        # If x axis is time, change ticks to formatted dates and times
        if x_param == 'time':
            xticks = axis.get_xticks()                             # Get the automatic xticks
            form = [argo_profile.format_jd(xt) for xt in xticks]   # Convert into formatted dates
            axis.set_xticklabels(form, rotation=90)                # Use dates as tick labels
        
        # Set axlims if given
        if ax_xlims is not None:
            axis.set_xlim(ax_xlims[0], ax_xlims[1])
        if ax_ylims is not None:
            axis.set_ylim(ax_ylims[0], ax_ylims[1])
        
        return to_ret
    
    """
    HARDCODED PLOT ROUTINES
    """
    def plot_isotherm_power_spectrum_avg(self, prof_indices='all',
                                         saveas=None, pt_range=None):
        # figure out prof nums
        if prof_indices == 'all': prof_indices = list(range(len(self.profiles)))
        
        # Plan not to display if saving
        if saveas != None:  plt.ioff()
        else:               plt.ion()
        
        # Make pt range if not given
        if pt_range == None:
            pt_range = np.arange(1.6, 1.801, .01)
            
        # Format plot text
        rc('text', usetex=True)
        rc('font', family='serif')
        plt.style.use('dark_background')
        
        # Get figure and axes, label with WMO_id and latest date
        f = plt.figure(figsize=(13,8))
        axis1 = plt.subplot(211)
        axis2 = plt.subplot(212)
        plt.suptitle("Float WMO\_ID: " + str(self.WMO_ID) + ", " +
                     "Latest day: "   + argo_profile.format_jd(max(self.julds)))
        plt.tight_layout()
        plt.subplots_adjust(top=0.88,bottom=0.185,left=0.065,right=0.99,hspace=0.2,wspace=0.2)
        
        self.axplot_avgspectrum(axis1, 'pt', pt_range, 'z', ax_ylims=[0.5, 10**6], ax_xlims=[.006, 0.5])
        self.axplot_isos(axis2, prof_indices, 'pt', pt_range, 'z')
        
        # Save and close if desired
        if saveas != None:
            f.savefig(saveas)
            plt.close(f)
        else: plt.show()
        
    def plot_steric_by_depth(self, prof_indices='all', saveas=None, bins=None):
        # figure out prof nums
        if prof_indices == 'all': prof_indices = list(range(len(self.profiles)))
        
        # Plan not to display if saving
        if saveas != None:  plt.ioff()
        else:               plt.ion()
        
        # Format plot text
        rc('text', usetex=True)
        rc('font', family='serif')
        plt.style.use('dark_background')
    
        # Get figure and axes, label with WMO_id and latest date
        f = plt.figure(figsize=(13,8))
        axis = plt.subplot(111)
        plt.suptitle("Float WMO\_ID: " + str(self.WMO_ID) + ", " +
                     "Latest day: "   + argo_profile.format_jd(self.julds[prof_indices[-1]]))
        
        self.steric_by_depth(axis, prof_indices)
        
        # Save and close if desired
        if saveas != None:
            f.savefig(saveas)
            plt.close(f)
        else: plt.show()      
        
    def plot_isotherms_collapsed(self, pt_range=None, prof_indices='all', saveas=None):
        """
        
        """
        # figure out prof nums
        if prof_indices == 'all': prof_indices = list(range(len(self.profiles)))
        
        # Plan not to display if saving
        if saveas != None:  plt.ioff()
        else:               plt.ion()
        
        # Make pt range if not given
        if pt_range == None:
            pt_range = np.arange(1.5, 1.901, .02)
    
        # Format plot text
        rc('text', usetex=True)
        rc('font', family='serif')
        plt.style.use('dark_background')
    
        # Get figure and axes, label with WMO_id and latest date
        f = plt.figure(figsize=(13,8))
        axis = plt.subplot(111)
        plt.suptitle("Float WMO\_ID: " + str(self.WMO_ID) + ", " +
                     "Latest day: "   + argo_profile.format_jd(self.julds[prof_indices[-1]]))
        plt.subplots_adjust(top=0.88,bottom=0.185,left=0.065,right=0.99,hspace=0.2,wspace=0.2)
        self.axplot_isos(axis, prof_indices, 'pt', pt_range, 'z')
        
        # Save and close if desired
        if saveas != None:
            f.savefig(saveas)
            plt.close(f)
        else: plt.show()         
         
    def plot_waterfall(self, pt_range=None, saveas=None):
        """
        
        """
        # Plan not to display if saving
        if saveas != None:  plt.ioff()
        else:               plt.ion()
        
        # Make pt range if not given
        if pt_range == None:
            pt_range = np.arange(1.2, 2.201, .01)
    
        # Format plot text
        rc('text', usetex=True)
        rc('font', family='serif')
        plt.style.use('dark_background')
    
        # Get figure and axes, label with WMO_id and latest date
        f = plt.figure(figsize=(13,8))
        axis = plt.subplot(111)
        plt.suptitle("Float WMO\_ID: " + str(self.WMO_ID) + ", " +
                     "Latest day: "   + argo_profile.format_jd(max(self.julds)))
        plt.subplots_adjust(top=0.88,bottom=0.185,left=0.065,right=0.99,hspace=0.2,wspace=0.2)
        self.axplot_spectrogram(axis, 'pt', pt_range, 'z')
        
        # Save and close if desired
        if saveas != None:
            f.savefig(saveas)
            plt.close(f)
        else: plt.show()
            
    def plot_pt_vs_SA(self, pt_range=[1.5,2.1], SA_range=[35.015, 35.07],
                      prof_indices='all',saveas=None):
        """
        
        """
        # figure out prof nums
        if prof_indices == 'all': prof_indices = list(range(len(self.profiles)))
        
        # Plan not to display if saving
        if saveas != None:  plt.ioff()
        else:               plt.ion()
    
        # Format plot text
        rc('text', usetex=True)
        rc('font', family='serif')
        plt.style.use('dark_background')
        
        # Get figure and axes, label with WMO_id and latest date
        f = plt.figure(figsize=(13,8))
        axis = plt.subplot(111)
        plt.suptitle("Float WMO\_ID: " + str(self.WMO_ID) + ", " +
                     "Latest day: "   + argo_profile.format_jd(self.julds[prof_indices[-1]]))
        plt.subplots_adjust(top=0.88,bottom=0.185,left=0.065,right=0.99,hspace=0.2,wspace=0.2)
        self.axplot_prof(axis, prof_indices, 'time_all', 'SA', 'pt',
                         cbar=True, ax_ylims = pt_range, ax_xlims=SA_range)
    
        # Save and close if desired
        if saveas != None:
            f.savefig(saveas)
            plt.close(f)
        else:
            plt.show()

    def plot_iso_depths(self, contour_levels=np.arange(1.5, 1.901, .02),
                        prof_indices='all', saveas=None, param='pt'):
        """
        
        """
        # figure out prof nums
        if prof_indices == 'all': prof_indices = list(range(len(self.profiles)))
        
        # Plan not to display if saving
        if saveas != None:  plt.ioff()
        else:               plt.ion()
        
        # Figure out spacing
        delta = abs(contour_levels[-1] - contour_levels[-2]) 
    
        # Format plot text
        rc('text', usetex=True)
        rc('font', family='serif')
        plt.style.use('dark_background')
        
        # Get figure and axes, label with WMO_id and latest date
        f = plt.figure(figsize=(13,8))
        axis = plt.subplot(111)
        plt.suptitle("Float WMO\_ID: " + str(self.WMO_ID) + ", " +
                     "Latest day: "   + argo_profile.format_jd(self.julds[prof_indices[-1]]) + ", " +
                     "Param is " + argo_profile.attr_to_name(param) + ", " +
                     r"$\Delta $" + param + " = " + str(delta))
        plt.subplots_adjust(top=0.88,bottom=0.185,left=0.065,right=0.95,hspace=0.2,wspace=0.2)
        self.axplot_iso_diffs(axis, prof_indices, param, contour_levels, cbar=True)
    
        # Save and close if desired
        if saveas != None:
            f.savefig(saveas)
            plt.close(f)
        else:
            plt.show()
    
    def plot_isotherms(self, contour_levels=np.arange(1.5, 1.901, .02),
                       prof_indices='all', saveas=None):
        """
        
        """
        # figure out prof nums
        if prof_indices == 'all': prof_indices = list(range(len(self.profiles)))
        
        # Plan not to display if saving
        if saveas != None:  plt.ioff()
        else:               plt.ion()
    
        # Format plot text
        rc('text', usetex=True)
        rc('font', family='serif')
        plt.style.use('dark_background')
        
        # Get figure and axes, label with WMO_id and latest date
        f = plt.figure(figsize=(13,8))
        axis = plt.subplot(111)
        plt.suptitle("Float WMO\_ID: " + str(self.WMO_ID) + ", " +
                     "Latest day: "   + argo_profile.format_jd(self.julds[prof_indices[-1]]))
        plt.subplots_adjust(top=0.88,bottom=0.185,left=0.065,right=0.95,hspace=0.2,wspace=0.2)
        self.axplot_prof(axis, prof_indices, 'isotherms', 'time', 'z',
                         contour_levels=contour_levels, cbar=False)
        axis.set_ylim(-5800, -4200)
    
        # Save and close if desired
        if saveas != None:
            f.savefig(saveas)
            plt.close(f)
        else:
            plt.show()

    def plot_depthvals(self, prof_indices='all', saveas=None, 
                       depths = [-4300, -4900, -5500], delta=True):
        """
        
        """
        # figure out prof nums
        if prof_indices == 'all': prof_indices = list(range(len(self.profiles)))
        
        # Plan not to display if saving
        if saveas != None:  plt.ioff()
        else:               plt.ion()
    
        # Format plot text
        rc('text', usetex=True)
        rc('font', family='serif')
        plt.style.use('dark_background')
        
        # Get figure and axes, label with WMO_id and latest date
        f = plt.figure(figsize=(13,8))
        gs = gridspec.GridSpec(2*len(depths), 1, figure=f)
        plt.suptitle("Float WMO\_ID: " + str(self.WMO_ID) + ", " +
                     "Latest day: "   + argo_profile.format_jd(self.julds[prof_indices[-1]]))
        plt.subplots_adjust(top=0.945,bottom=0.16,left=0.085,right=0.975,hspace=0.02,wspace=0.19)
        
        mid_i = sum(range(len(depths)))/len(depths)
        for i, depth in enumerate(depths):
            pt_ax = plt.subplot(gs[i,:])
            SA_ax = plt.subplot(gs[i+len(depths),:])
            
            self.axplot_depthval(pt_ax, prof_indices, depth, 'pt', delta=delta,
                                 ax_xlabel=False, ax_xticklabel=False,
                                 ax_ylabel=(i==mid_i))
            self.axplot_depthval(SA_ax, prof_indices, depth, 'SA', delta=delta, 
                                 ax_xlabel     = (i+1 == len(depths)),
                                 ax_xticklabel = (i+1 == len(depths)),
                                 ax_ylabel=(i==mid_i))
    
        # Save and close if desired
        if saveas != None:
            f.savefig(saveas)
            plt.close(f)
        else:
            plt.show()
        
    def plot_pt_sa_time_all(self, prof_indices='all', saveas=None):
        """
         
        """
        # figure out prof nums
        if prof_indices == 'all': prof_indices = list(range(len(self.profiles)))
            
        # Plan not to display if saving
        if saveas != None:  plt.ioff()
        else:               plt.ion()
            
        # Format plot text
        rc('text', usetex=True)
        rc('font', family='serif')
        plt.style.use('dark_background')
        
        # Get figure and axes, label with WMO_id and latest date
        f = plt.figure(figsize=(13,8))
        gs = gridspec.GridSpec(1, 2, figure=f)
        pt_ax = plt.subplot(gs[:, 0])
        SA_ax = plt.subplot(gs[:, 1])
        plt.suptitle("Float WMO\_ID: " + str(self.WMO_ID) + ", " +
                     "Latest day: "   + argo_profile.format_jd(self.julds[prof_indices[-1]]))
        plt.tight_layout()
        f.subplots_adjust(top=0.95,bottom=0.07,left=0.08, right=0.920,wspace=0.15)
        self.axplot_prof(pt_ax, prof_indices, 'time_all', 'pt', 'z',
                         cbar=False, deltax=True, ax_ylims = [-6000, -4000],
                         ax_xlims=[-.1, .1])
        self.axplot_prof(SA_ax, prof_indices, 'time_all', 'SA', 'z',
                         cbar=True, deltax=True, ax_ylims = [-6000, -4000],
                         ax_xlims=[-.01, .01])
        
        # Save and close if desired
        if saveas != None:
            f.savefig(saveas)
            plt.close(f)
        else:
            plt.show()             
    
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
        plt.suptitle("Float WMO\_ID: " + str(self.WMO_ID) + ", " +
                     "Latest day: "   + argo_profile.format_jd(self.julds[prof_indices[-1]]))
        
        # Plot all axis labels, adjust size and shape for ~ok~ saved formatting
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

    def plot_profiles_with_contours(self, prof_indices='all', saveas=None):
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
            saveas       - 
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
    
        # Format plot text
        rc('text', usetex=True)
        rc('font', family='serif')
        plt.style.use('dark_background')
        
        # Get figure and axes, label with WMO_id and latest date
        f = plt.figure(figsize=(13,8))
        gs = gridspec.GridSpec(3, 3, figure=f)
        
        map_ax = plt.subplot(gs[ :   , 0 ])
        
        season_ax_top = plt.subplot(gs[0 , 1])
        season_ax_mid = plt.subplot(gs[1 , 1])
        season_ax_bot = plt.subplot(gs[2 , 1])
        
        isotherm_ax_top = plt.subplot(gs[0 , 2])
        isotherm_ax_mid = plt.subplot(gs[1 , 2])
        isotherm_ax_bot = plt.subplot(gs[2 , 2])
        
        plt.suptitle("Float WMO\_ID: " + str(self.WMO_ID) + ", " +
                     "Latest day: "   + argo_profile.format_jd(self.julds[prof_indices[-1]]))
        f.subplots_adjust(top=0.935,bottom=0.20,left=0.055,right=0.940,hspace=0.15,wspace=0.4)
    
        
        # MAP axis : Location over time
        map = self.aviso_map_plot(days[-1], map_ax, extracolors='white')              # Plot the interpolated SLA values
        x, y = map(lons, lats)                                   # Get map float locations
        map.plot(x,y, label='Float location', color='black')     # Plot float locations
        map.scatter(x[-1], y[-1], color='teal')                  # And current float location as a dot
        
        
        
        # SEASON axis : potential temp vs absolute salinity by season
        
        # Step 1. Plot
        self.axplot_prof(season_ax_top, prof_indices, 'seasonal', 'SA', 'pt',
                         cbar=False, ax_ylims = [18, 30], ax_xlims=[36.5, 38])
        self.axplot_prof(season_ax_mid, prof_indices, 'seasonal', 'SA', 'pt',
                         cbar=False, ax_ylims = [0,  30], ax_xlims=[34.5, 38])
        sb = self.axplot_prof(season_ax_bot, prof_indices, 'seasonal', 'SA', 'pt',
                              cbar=False, ax_ylims = [1.5,  3], ax_xlims=[35.0, 35.15])
        
        # Step 2: Share x axis labels
        season_ax_top.set_xlabel('')
        season_ax_mid.set_xlabel('')
        
        season_ax_top.set_ylabel('')
        season_ax_bot.set_ylabel('')
        
        # Add colorbar
        cbar = f.colorbar(sb, ax=[season_ax_top, season_ax_mid, season_ax_bot], 
                          ticks=np.linspace(0, 1, num=13))
        cbar.ax.set_yticklabels(['July', 'August', 'September',
                                 'October',  'November', 'December',
                                 'January', 'February', 'March',
                                 'April', 'May', 'June', 'July'])
        #cbar.set_label('Month', rotation=270)
        
        # ISOTHERM axis : Time vs. Depth vs. Potential Temperature
        
        # Step 1. Plot
        it, its = self.axplot_prof(isotherm_ax_top, prof_indices, 'isotherms', 'time', 'z',
                                   vmin=0, vmax=30, cbar=False, ax_ylims=[-500, 0])
        im, ims = self.axplot_prof(isotherm_ax_mid, prof_indices, 'isotherms', 'time', 'z',
                                   vmin=0, vmax=30, cbar=False, ax_ylims=[-2000, -500])
        ib, ibs = self.axplot_prof(isotherm_ax_bot, prof_indices, 'isotherms', 'time', 'z',
                                   vmin=0, vmax=30, cbar=False, ax_ylims=[-6000, -2000])
        
        # Step 2: Share x axis labels and remove excess ticks
        isotherm_ax_top.set_xlabel('')
        isotherm_ax_mid.set_xlabel('')
        isotherm_ax_top.set_ylabel('')
        isotherm_ax_bot.set_ylabel('')
        
        isotherm_ax_top.get_xaxis().set_ticks([])
        isotherm_ax_mid.get_xaxis().set_ticks([])
        
        # Add colorbar
        cbar = f.colorbar(ib, ax=[isotherm_ax_top, isotherm_ax_mid, isotherm_ax_bot])
        cbar.set_label('Potential Temperature (C)', rotation=270, labelpad=15)
        cbar.add_lines(ibs)
        
        # Save and close if desired
        if saveas != None:
            f.savefig(saveas)
            plt.close(f)
        else:
            plt.show()             

    def plot_profile_with_contour_latest_interactive(self):
        """
        as plot_profiles_with_contours but plot all indices and display interactive.
        also, collapse the 3 plots with contours into 1. 
        """
        prof_indices = list(range(len(self.profiles)))
        
        # Get profiles information
        lats       = [self.profiles[i].latitude         for i in prof_indices]
        lons       = [self.profiles[i].longitude        for i in prof_indices]
        days       = [self.julds[i]                     for i in prof_indices]
    
        # Format plot text
        rc('text', usetex=True)
        rc('font', family='serif')
        plt.style.use('dark_background')
        
        # Get figure and axes, label with WMO_id and latest date
        f = plt.figure(figsize=(13,8))
        gs = gridspec.GridSpec(3, 3, figure=f)
        map_ax = plt.subplot(gs[ :   , 0 ])
        season_ax = plt.subplot(gs[ : , 1])
        isotherm_ax = plt.subplot(gs[ : , 2])
        plt.suptitle("Float WMO\_ID: " + str(self.WMO_ID) + ", " +
                     "Latest day: "   + argo_profile.format_jd(self.julds[prof_indices[-1]]))
        f.subplots_adjust(top=0.935,bottom=0.20,left=0.055,right=0.940,hspace=0.15,wspace=0.4)
    
        # MAP axis : Location over time
        map = self.aviso_map_plot(days[-1], map_ax, extracolors='white')     # Plot the interpolated SLA values
        x, y = map(lons, lats)                                               # Get map float locations
        map.plot(x,y, label='Float location', color='black')                 # Plot float locations
        map.scatter(x[-1], y[-1], color='teal')                              # And current float location as a dot
    
        # SEASON axis : potential temp vs absolute salinity by season
        sm = self.axplot_prof(season_ax, prof_indices, 'seasonal', 'SA', 'pt',
                              cbar=False, ax_ylims = [0,  30], ax_xlims=[34.5, 38])
        cbar = f.colorbar(sm, ax=season_ax, 
                          ticks=np.linspace(0, 1, num=13))
        cbar.ax.set_yticklabels(['July', 'August', 'September',
                                 'October',  'November', 'December',
                                 'January', 'February', 'March',
                                 'April', 'May', 'June', 'July'])
        
        # ISOTHERM axis : Time vs. Depth vs. Potential Temperature
        it, its = self.axplot_prof(isotherm_ax, prof_indices, 'isotherms', 'time', 'z',
                                   vmin=0, vmax=30, cbar=False, ax_ylims=[-6000, 0])
        cbar = f.colorbar(it, ax=[isotherm_ax])
        cbar.set_label('Potential Temperature (C)', rotation=270, labelpad=15)
        cbar.add_lines(its)
        
        plt.show()
        
        
    """
    SERIES PLOTTING FUNCTIONS
    """

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
    
    def plot_profs(self, deltas = False, contours = False, pt_sa = False, pt_vs_sa = False,
                   savedir='/home/cassandra/docs/argo/movies/float/rel_weekly/',
                   WMO_ID_dir=True, endplot=True):
        """
        Plot all the profiles which have aviso data overlap
        """
        # if deltas, modify savedir to be another directory
        if     deltas: savedir = savedir[:-1]+'_deltas/'
        elif contours: savedir = savedir[:-1]+'_contours/'
        elif    pt_sa: savedir = savedir[:-1]+'_pt_sa/'
        elif pt_vs_sa: savedir = savedir[:-1]+'_pt_vs_sa/'
        else:          savedir = savedir[:-1]+'_default/'
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
            elif contours:
                self.plot_profiles_with_contours(prof_indices = indices_to_use, 
                                                 saveas       = (savedir+'f{:0>3d}.png'.format(i)))
            elif pt_sa:
                self.plot_pt_sa_time_all(prof_indices = indices_to_use, 
                                         saveas       = (savedir+'f{:0>3d}.png'.format(i)))
            elif pt_vs_sa:
                self.plot_pt_vs_SA(prof_indices = indices_to_use, 
                                   saveas       = (savedir+'f{:0>3d}.png'.format(i)))
            else:
                self.plot_profiles(prof_indices = indices_to_use, 
                                   saveas       = (savedir+'f{:0>3d}.png'.format(i)))
        if endplot:
            if deltas:
                self.plot_profiles_with_deltas()
            elif contours:
                self.plot_profiles_with_contours()
            elif pt_sa:
                self.plot_pt_sa_time_all()
            elif pt_vs_sa:
                self.plot_pt_vs_SA()
            else:
                self.plot_profiles()




def main():
    """
    Make movies for every deep argo float in the northwest atlantic
    """    
    start_time = time.time()
    for WMO_ID in  [4902322, 4902323, 4902324]: #[4902321, 4902322, 4902323, 4902324, 4902325, 4902326]:
        
        print("Plotting float with WMO-ID " + str(WMO_ID))
        print("Time is " + str(int((time.time()-start_time)/60)) + " minutes.")
        sys.stdout.flush()
        
        #
        # Week interpolated Aviso data
        #
        af_weekly = ArgoFloat(WMO_ID, argo_dir="/data/deep_argo_data/nc/",
                              aviso_dir = '/data/aviso_data/nrt/weekly/')
        #af_weekly.plot_depthvals()
        #af_weekly.plot_pt_vs_SA()
        #af_weekly.plot_profs(pt_sa=True)
        #af_weekly.plot_iso_depths()
        #af_weekly.plot_profs(pt_vs_sa=True)
        #af_weekly.plot_profile_with_contour_latest_interactive()
        #af_weekly.plot_prof_relevant()
        #af_weekly.plot_isotherms_collapsed()
        #af_weekly.plot_waterfall()
        #af_weekly.plot_steric_by_depth()
        af_weekly.plot_isotherm_power_spectrum_avg()
        gc.collect()
        
        #
        # Month interpolated Aviso data
        #
        #af_monthly = ArgoFloat(WMO_ID, argo_dir="/data/deep_argo_data/nc/",
        #                       aviso_dir = '/data/aviso_data/monthly_mean/')
        #af_monthly.plot_prof_relevant(deltas=True, savedir='/home/cassandra/docs/argo/movies/float/rel_monthly/')
        #af_monthly.plot_prof_relevant(savedir='/home/cassandra/docs/argo/movies/float/rel_monthly/')
        
        
if __name__ == '__main__':
    main()