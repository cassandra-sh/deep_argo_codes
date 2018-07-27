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
import numpy as np

def num_only(string):
    return "".join(_ for _ in string if _ in "1234567890")

class ArgoFloat:
    """
    Contains all profiles for a given argo float, and an aviso interpolator
    clipped to the location of the float.
    """
    def __init__(self, float_number, **kwargs):
        """
        
        @params 
            float_number
        **kwargs
            argo_dir
            lim
            
        """
        # Get files
        self.argo_dir     = kwargs.get('argo_dir', "/data/argo_data/nc/")
        self.float_number = float_number
        obs_files = glob.glob(self.argo_dir+"*"+str(float_number)+"*.nc")
        
        # Read files as argo profiles and get observation numbers
        self.obs_nums = [int(num_only(f.split('_')[-1])) for f in obs_files]
        self.profiles = [argo_profile.from_nc_ubu(f) for f in obs_files]
        
        # Sort by observation number
        lim = kwargs.get('lim', 30)
        self.profiles = [p for _,p in sorted(zip(self.obs_nums, self.profiles))][:lim]
        self.obs_nums = sorted(self.obs_nums)[:lim]
        
        # Get the first location
        self.lon0 = self.profiles[0].longitude
        self.lat0 = self.profiles[0].latitude
        
        # Get the average P/T/S profile
        self.pres_avg = np.arange(0, 6000.05, 0.5)
        psal_interps = [prof.interp_psal(self.pres_avg) for prof in self.profiles]
        temp_interps = [prof.interp_temp(self.pres_avg) for prof in self.profiles]
        self.psal_avg = np.sum(psal_interps, axis=1)/len(self.profiles)
        self.temp_avg = np.sum(temp_interps, axis=1)/len(self.profiles)
        
        # Get an aviso interpolator within a degree range of this location
        self.AIbox = [self.lon0-10.0, self.lon0+10.0,
                      self.lat0-10.0, self.lat0+10.0]
        self.AI = aviso_interp.AvisoInterpolator(box=self.AIbox, irregular=False,
                                                 limit=24, verbose=False)
    
    
    def aviso_plot(self, date, axis):
        """
        Plot the sea level anomaly for a given day on a given axis
        """
        # 1. Plot the basemap
        plt.sca(axis)
        map = Basemap(llcrnrlat = self.lat0 - 10.0, urcrnrlat = self.lat0 + 10.0,
                      llcrnrlon = self.lon0 - 10.0, urcrnrlon = self.lon0 + 10.0)
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
        pass      
        
        
    
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
        ax1.plot(days, aviso_sla)                             # Plot the aviso SLA
        ax1.plot(days, steric_sla)                            # Plot the steric SLA
        xticks = ax1.get_xticks()                             # Get the automatic xticks
        form = [argo_profile.format_jd(xt) for xt in xticks]  # Convert into formatted dates
        ax1.set_xticklabels(form, rotation=90)                # Use dates as tick labels
        ax1.grid(True)                                        # Add grid
        
        
        # AXIS 2 : Pressure vs. Temperature for floats
        for i in prof_indices:
            ax2.scatter(self.profiles[i].pressure,            # Plot the P/T profiles
                        self.profiles[i].temperature, s=1)    # Use very small markers
        ax2.invert_yaxis()                                    # Invert pressure axis to descending
        ax2.grid(True)                                        # Add grid
        
        
        # AXIS 3 : Location over time
        map = self.aviso_plot(days[-1], ax3)                  # Plot the interpolated SLA values
        x, y = map(lons, lats)                                # Get map float locations
        map.plot(x,y, label='Float location', color='black')  # Plot float locations
        
        
        # Save and close if desired
        if saveas != None:
            f.savefig(saveas)
            plt.close(f)
        else:
            plt.show()
    
    def plot_prof_all(self, savedir='/home/cassandra/docs/argo/movies/float/'):
        """
        Plot all of the profiles for this float
        """
        indices = list(range(1, len(self.profiles)))
        for i in indices:
            print('Plotting index', i)
            saveas = savedir+'f{:0>3d}.png'.format(i)
            self.plot_profiles(prof_indices=indices[:i+1], saveas=saveas)
            
    def plot_prof_relevant(self, savedir='/home/cassandra/docs/argo/movies/float/'):
        """
        Plot all the profiles which have aviso data overlap
        """
        pass




def main():
    """
    
    """
    float_number = 4902326
    AF = ArgoFloat(float_number, argo_dir="/data/deep_argo_data/nc/", lim=26)
    AF.plotrange()
        
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
# display_map() - originally a test method 
"""
    def display_map(self, savedir='/home/cassandra/docs/argo/movies/float/'):
        start_time = time.time()
        
        # Report code start
        print("Starting code by prepping interp inputs. Time is " + str(int(time.time() - start_time)))
        sys.stdout.flush()
        
        # Get a date range
        jd = [prof.get_julian_day() for prof in self.profiles]
        
        # Get a range of lat/lon values
        lons = np.arange(self.AIbox[0], self.AIbox[1], 0.05)
        lats = np.arange(self.AIbox[2], self.AIbox[3], 0.05)
        
        # Fix the lons to be from 0 to 360
        lons[lons<0] = lons[lons<0]+360
        
        # Turn these into a grid to interpolate on
        dd, la, lo = np.meshgrid(jd, lats, lons, indexing='ij')
        
        # Report that we're about to interpolate
        print("About to interpolate. Time is " + str(int(time.time() - start_time)))
        sys.stdout.flush()
        
        # Get the interpolator sea level anomaly (sla) values for these points
        vals = self.AI.interpolate(dd, la, lo)
        
        # Report done interpolating
        print("Done interpolating. Time is " + str(int(time.time() - start_time)))
        sys.stdout.flush()
        
        
        # Get profile locations and sla values
        prof_lats, prof_lons, prof_sla = [], [], []
        for prof in self.profiles:
            prof_lats.append(prof.latitude)
            prof_lons.append(prof.longitude)
            
            lon = prof.longitude
            if lon < 0: lon = lon + 360
            
            prof_sla.append(self.AI.interpolate(prof.get_julian_day(), prof.latitude, lon))
        
        
        # Wait for user input
        print("Ready to plot. Time is " + str(int(time.time() - start_time)))
        sys.stdout.flush()
        input('Press enter to plot:')
                
        # Report that we're about to plot
        print("Plotting. Time is " + str(int(time.time() - start_time)))
        sys.stdout.flush()
        
        # Get figure
        f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
        
        # Plot SLA stuff
        plt.sca(ax1)
        ax1.plot(jd[0], prof_sla[0])
        ax1.set_xlabel('Julian Day')
        ax1.set_ylabel("Sea Level Anomaly (m)")
        
        # Plot PT stuff 
        plt.sca(ax2)
        ax2.scatter(self.profiles[0].pressure, self.profiles[0].temperature, s=5)
        ax2.set_ylabel("Pressure")
        ax2.set_xlabel("Temperature (in-situ)")
        ax2.invert_yaxis()
        
        # Prep a basemap to plot location on
        plt.sca(ax3)
        map = Basemap(llcrnrlat = prof_lats[0] - 30.0, urcrnrlat = prof_lats[0]  + 30.0,
                      llcrnrlon = prof_lons[0]- 40.0, urcrnrlon = prof_lons[0] + 40.0)
        map.drawmapboundary()
        map.fillcontinents()
        map.drawcoastlines()
        parallels = np.arange(-80.,81,10.)
        map.drawparallels(parallels, labels=[False, True, False, False])
        meridians = np.arange(10.,351.,20.)
        map.drawmeridians(meridians, labels=[False, False, False, True])
        
        
        # Plot the sea level anomaly for the first time
        valmin = np.nanmin(vals)
        valmax = np.nanmax(vals)
        map.pcolor(lons, lats, vals[0], cmap='coolwarm', latlon=True, 
                   vmin=valmin, vmax=valmax)
        
        # Plot the float location for the first time
        x, y = map(prof_lons[0], prof_lats[0])
        map.plot(x,y, label='Float location')
        
        # Plot some information and colorbar
        cbar = plt.colorbar(orientation='horizontal')
        cbar.set_label('Sea Level Anomaly (m)')
        plt.title(str(self.argo_dir + " float num " + str(self.float_number)))
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.tight_layout()
        plt.show()
        
        # Wait 10 seconds for formatting plot, etc
        print("Waiting to begin walk. Time is " + str(int(time.time() - start_time)))
        sys.stdout.flush()
        plt.pause(20)
        
        # Save image
        f.savefig(savedir+'frame0.png')
                
        # In a loop, update the anomaly, the float location, wait a second, repeat
        # also add the PT prof
        for i in range(1, len(jd)):
            # Report
            print("Update num "+str(i)+"! Time is " + str(int(time.time() - start_time)))
            sys.stdout.flush()
            
            # SLA stuff
            plt.sca(ax1)
            ax1.plot(jd[i-1:i+1], prof_sla[i-1:i+1], color='black')
            
            # PT stuff
            plt.sca(ax2)
            ax2.scatter(self.profiles[i].temperature, self.profiles[i].pressure, s=5, zorder=i)
            
            # BASEMAP stuf
            plt.sca(ax3)
            
            # Plot the sea level anomaly for the interpolated area
            map.pcolor(lons, lats, vals[i], cmap='coolwarm', latlon=True, zorder=i, 
                       vmin=valmin, vmax=valmax)
            
            # Plot the float location
            x, y = map(prof_lons[:i+1], prof_lats[:i+1])
            map.plot(x,y, label='Float location', zorder=i+1)
            
            # Redraw boundaries and coastlines
            map.drawmapboundary()
            map.fillcontinents()
            map.drawcoastlines()
            parallels = np.arange(-80.,81,10.)
            map.drawparallels(parallels, labels=[False, True, False, False])
            meridians = np.arange(10.,351.,20.)
            map.drawmeridians(meridians, labels=[False, False, False, True])
            
            # Save image
            f.savefig(savedir+'frame'+str(i)+'.png')
            
            
        
        # Report end time
        print("Done plotting! Time is " + str(int(time.time() - start_time)))
        sys.stdout.flush()
"""