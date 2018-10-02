#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 14:15:08 2018

@author: cassandra
"""
import numpy        as np
import argo_float   as af
import argo_profile as ap
import aviso_interp

import matplotlib
import matplotlib.gridspec as     gridspec
import matplotlib.pyplot   as     plt
from matplotlib.pyplot     import cm
from matplotlib            import rc
from mpl_toolkits.basemap  import Basemap
import matplotlib.patheffects as PathEffects
import matplotlib.patches as mpatches

class FloatComparer:
    """
    Compare some ArgoFloat objects at different depths
    """
    
    def __init__(self, wmo_ids, day_range=None, float_colors=None, 
                 aviso_dir = '/data/aviso_data/nrt/weekly/', **kwargs):
        """
        Initialize the FloatComparer with a list of WMO_IDs and **kwargs to pass
        to the ArgoFloat initializer. 
        
        @params
            wmo_ids
            day_range = None
            float_colors
            aviso_dir
            
        for constructing the AvisoInterpolator object
            hspace = 10 - horizontal buffer space between left and right plot 
                          boundaries and leftmost and rightmost float locations
            vspace = 10 - as hspace, but vertical buffer against topmost and 
                          bottommost float locations. 
        """
        # Build ArgoFloat objects from the given wmo_ids, and get some basic info ready
        self.wmo_ids = wmo_ids
        self.floats = [af.ArgoFloat(wmo_id, **kwargs) for wmo_id in wmo_ids]
        self.n_floats = len(self.floats)
        self.synth_profiles = {}
        if day_range is not None:
            self.prep_interp_profs(day_range)
            
        # Either adopt the given color range or pick one
        if float_colors is not None:
            self.float_colors = float_colors
        else:
            color=iter(cm.rainbow(np.linspace(0,1,self.n_floats)))
            self.float_colors = [next(color) for _ in range(self.n_floats)]
            #self.float_colors = ['white' for _ in range(self.n_floats)]
        self.float_colors=['red', 'green', 'blue', 'yellow']
            
        # Get minimum and maximum lons and lats for the aviso interp object
        self.max_lon = np.nanmax(np.nanmax([argofloat.lons for argofloat in self.floats]))
        self.min_lon = np.nanmin(np.nanmin([argofloat.lons for argofloat in self.floats]))
        
        self.max_lat = np.nanmax(np.nanmax([argofloat.lats for argofloat in self.floats]))
        self.min_lat = np.nanmin(np.nanmin([argofloat.lats for argofloat in self.floats]))
        
        # Make AvisoInterpolator object
        hspace = kwargs.get('hspace', 18.0)
        vspace = kwargs.get('vspace', 10.0)
        self.AIbox = [self.min_lon-hspace, self.max_lon+hspace,
                      self.min_lat-vspace, self.max_lat+vspace]
        self.AI = aviso_interp.AvisoInterpolator(box=self.AIbox, irregular=False,
                                                 limit=100, verbose=False,
                                                 aviso_dir = aviso_dir,
                                                 units = 'm')
    
    def any_match_days(self):
        """
        Return all days, sorted, which have aviso data and at least one argo float's data
        """
        return sorted(list(set().union(*[argofloat.match_days() for argofloat in self.floats])))
    
    def all_match_day_range(self):
        min_max = max(self.floats[0].match_days())
        max_min = min(self.floats[0].match_days())
        
        for argofloat in self.floats:
            days = argofloat.match_days()
            
            if min_max > max(days):
                min_max = max(days)
            
            if max_min < min(days):
                max_min = min(days)
        
        return max_min, min_max


    def aviso_map_plot(self, axis, date, hspace=16.0, vspace=10.0, extracolors='black'):
        """
        Plot a basemap to the given axis to display all the aviso data for this
        collection of floats, on the given day. Return the Basemap object.
        
        @params
            axis        - matplotlib axis to plot the basemap on
            date        - Julian day to get aviso data (interpolated) to
            hspace = 10 - horizontal buffer space between left and right plot 
                          boundaries and leftmost and rightmost float locations
            vspace = 10 - as hspace, but vertical buffer against topmost and 
                          bottommost float locations. 
            extracolors = 'black' - color to plot text and meridians/parallels in
        """ 
        # 1. Plot the basemap
        plt.sca(axis)
        map = Basemap(llcrnrlat = self.min_lat-vspace, urcrnrlat = self.max_lat+vspace,
                      llcrnrlon = self.min_lon-hspace, urcrnrlon = self.max_lon+hspace)
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
        #map.pcolor(lons, lats, vals[0], cmap='coolwarm',           
        #           latlon=True, vmin=self.AI.min, vmax=self.AI.max, zorder=1)
        #cbar = plt.colorbar()
        #cbar.set_label('Sea Level Anomaly (m)')
        
        return map
    
    def full_map_plot(self, axis, day_range, hspace=13.0, vspace=3.0, extracolors='black'):
        """
        Plot a map where all floats locations, present and past, are plotted and
        labeled by wmo_id, and aviso interpolated SLA data is underneath. 
        
        @params
            axis        - matplotlib axis to plot the basemap on
            day_range   - Day range to plot (last day in list is top value for location/parameter plot)
            hspace = 10 - horizontal buffer space between left and right plot 
                          boundaries and leftmost and rightmost float locations
            vspace = 10 - as hspace, but vertical buffer against topmost and 
                          bottommost float locations. 
            extracolors = 'black' - color to plot text and meridians/parallels in
        
        @returns the Basemap object
        """
        
        # 1. Draw map
        map = self.aviso_map_plot(axis, day_range[-1], hspace=hspace, 
                                  vspace=vspace, extracolors=extracolors)
        
        # 2. Plot locations and text for each float
        for i in range(self.n_floats):
            
            # a. Turn float interpolated lons and lats for this day into map coordinates
            x, y = map(self.floats[i].lon_interp(day_range), self.floats[i].lat_interp(day_range))
            
            # b. Plot, emphasizing the latest position as a colored dot
            map.plot(x, y, color='black', zorder=2)
            map.scatter(x[-1], y[-1]+2, color='black', s=15, zorder=3)
            
            # c. Label each float by wmo_id
            xt, yt = map(self.floats[i].lon_interp(day_range), self.floats[i].lat_interp(day_range)+4.0)
            axis.text(xt[-1], yt[-1], str(self.wmo_ids[i]), fontsize=18, rotation=30, color='black', zorder=4)
        
        return map
    
    
    def plot_map_only(self, day_range=None, saveas=None):
        """
        x axis: longitude
        y axis: heave
        z axis: time (darkest = recent past, lightest = distant past)
        
        day_range: up to given date
        plots: heave signals for each *profile* at the location measured
        
        maybe eventually fit some kind of curve but it depends what it looks 
        like and what kind of prior we can put on deep mode wave speed and
        wavenumber
        """
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        rc('text', usetex=True)
        rc('font', family='serif', size=24)
        
        if day_range is None:
            min_day, max_day = self.all_match_day_range()
            day_range = np.linspace(min_day, max_day, 27*(max_day-min_day)/365)
        day_range = sorted(day_range)
        
        f = plt.figure()
        map_axis = plt.subplot()
        self.full_map_plot(map_axis, day_range[:1], extracolors='black',
                           hspace=14, vspace=7)
        
        if saveas != None:
            f.savefig(saveas)
            plt.close(f)
        else:
            plt.show()
    
    def plot_heave_longitude(self, day_range=None, saveas=None):
        """
        x axis: longitude
        y axis: heave
        z axis: time (darkest = recent past, lightest = distant past)
        
        day_range: up to given date
        plots: heave signals for each *profile* at the location measured
        
        maybe eventually fit some kind of curve but it depends what it looks 
        like and what kind of prior we can put on deep mode wave speed and
        wavenumber
        """
        
        #1. For each Float, get the isopycnals and the means and standard deviations
        #   of the heave signal
        
        #2. Determine what data points will be plotted based on the day range
        
        #3. Color based on how many days have passed since the latest day
        
        #4. Plot
    
    def plot_bathymetry_position(self, day_range=None, saveas=None):
        """
        
        """
        # Generate day range if needed. Format to sorted just in case.
        if day_range is None:
            min_day, max_day = self.all_match_day_range()
            day_range = np.linspace(min_day, max_day, 27*(max_day-min_day)/365)
        day_range = sorted(day_range)
        # Format matplotlib as desired
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        rc('text', usetex=True)
        rc('font', family='serif', size=18)
        # PREPARE AXES    
        f = plt.figure(figsize=(13,8))
        gs = gridspec.GridSpec(2, self.n_floats, figure=f)
        map_axis = plt.subplot(gs[ 0, : ])
        self.full_map_plot(map_axis, day_range, extracolors='black')
        float_axes = [plt.subplot(gs[1,i]) for i in range(self.n_floats)]
        for i in range(len(self.floats)):
            prof_indices = self.floats[i].profs_to_date(day_range[-1])
            self.floats[i].axplot_isopycnal_bathymetry(float_axes[i], prof_indices)
            float_axes[i].set_title(self.wmo_ids[i])
        plt.suptitle("Latest day: "   + ap.format_jd(day_range[-1]))
        plt.subplots_adjust(top=0.962,bottom=0.113,left=0.076,right=0.97,hspace=0.136,wspace=0.364)
        # Save and close if desired
        if saveas != None:
            f.savefig(saveas)
            plt.close(f)
        else:
            plt.show()
    
    
    def plot_isotherms_together(self, day_range=None, saveas=None, contour_levels=[ 1.52, 1.65, 1.82]):
        # Plan not to display if saving, by turning interactive mode off (or on if planning to show)
        if saveas != None:  plt.ioff()
        else:               plt.ion()

        
        #contour_levels = list(np.arange(1.50, 1.90, 0.02))
        #contour_levels=[ 1.52, 1.54, 1.56, 1.58, 1.60]#, 1.65, 1.82]
        
        # Format plot text
        rc('text', usetex=True)
        rc('font', family='serif')
        plt.style.use('dark_background')
        
        # Generate day range if needed. Format to sorted just in case.
        if day_range is None:
            min_day, max_day = self.all_match_day_range()
            day_range = np.linspace(min_day, max_day, 27*(max_day-min_day)/365)
        day_range = sorted(day_range)
        
        # Pre figure and axis
        f = plt.figure(figsize=(13,8))
        gs = gridspec.GridSpec(2, 1, figure=f)
        map_axis = plt.subplot(gs[0, :])
        
        self.full_map_plot(map_axis, day_range, extracolors='white')
        
        axis = plt.subplot(gs[1, :])
        patches = []
        for j in range(self.n_floats):
            prof_indices = self.floats[j].profs_to_date(day_range[-1])
            
            xvals = np.array([self.floats[j].julds[i] for i in prof_indices])
            
            depthvals = np.array([self.floats[j].profiles[i].z_interp(self.floats[j].pres_range) for i in prof_indices])
            yvals = np.nanmean(depthvals, axis=0)
            
            zvals = np.array([self.floats[j].profiles[i].pt_interp(self.floats[j].pres_range) for i in prof_indices]).T
            cs = axis.contour(xvals, yvals, zvals,
                              levels=sorted(contour_levels),
                              colors=self.float_colors[j],
                              extent=[np.nanmin(xvals), np.nanmax(xvals),
                                      np.nanmin(yvals), np.nanmax(yvals)],
                              vmax = max(contour_levels)+1,
                              vmin = min(contour_levels)-1)
            clabels = plt.clabel(cs, color=self.float_colors[j])        
            patches.append(mpatches.Patch(color=self.float_colors[j], 
                                          label=r'Float WMO\_ID: ' + str(self.wmo_ids[j])))
            for l in clabels:
                l.set_rotation(0)
                l.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='k')])
            
        axis.set_xlim(np.nanmin(xvals), np.nanmax(xvals))    
        axis.set_ylim(-5200, -4200)
                
        xticks = axis.get_xticks() 
        form = [ap.format_jd(xt) for xt in xticks]     # Convert into formatted dates
        axis.set_xticklabels(form, rotation=90)
        
        
        plt.legend(handles=patches)
        plt.subplots_adjust(top=0.96,bottom=0.2,left=0.055,right=0.96,hspace=0.36,wspace=0.32)
        plt.suptitle("Latest day: "   + ap.format_jd(day_range[-1]))
        
        
        # Save and close if desired
        if saveas != None:
            f.savefig(saveas)
            plt.close(f)
        else:
            plt.show()
               
    def plot_profiles_compare(self, day_range, saveas=None):
        """
        Mostly hardcoded plot routine for comparing potential temperature 
        versus Absolute salinity between floats over time
        """
        # Plan not to display if saving, by turning interactive mode off (or on if planning to show)
        if saveas != None:  plt.ioff()
        else:               plt.ion()
        
        # Format plot text
        rc('text', usetex=True)
        rc('font', family='serif')
        plt.style.use('dark_background')
        
        # Generate day range if needed. Format to sorted just in case.
        if day_range is None:
            day_range = self.any_match_days()
        day_range = sorted(day_range)
        
        # PREPARE AXES    
        f = plt.figure(figsize=(13,8))
        gs = gridspec.GridSpec(5, self.n_floats, figure=f)
        
        map_axis = plt.subplot(gs[ :2, : ])
        float_tops = [plt.subplot(gs[2,i]) for i in range(self.n_floats)]
        float_mids = [plt.subplot(gs[3,i]) for i in range(self.n_floats)]
        float_bots = [plt.subplot(gs[4,i]) for i in range(self.n_floats)]
        
        plt.subplots_adjust(top=0.97,bottom=0.08,left=0.055,right=0.96,hspace=0.36,wspace=0.32)
        plt.suptitle("Latest day: "   + ap.format_jd(day_range[-1]))
        
        # Step 1: Plot map
        self.full_map_plot(map_axis, day_range, extracolors='white')
        
        # Step 2: Plot the profiles
        sts, sms, sbs = [], [], []
        for i in range(self.n_floats):
            prof_indices = self.floats[i].profs_to_date(day_range[-1])
            st = self.floats[i].axplot_prof(float_tops[i], prof_indices, 'seasonal', 'SA', 'pt',
                                            cbar=False, ax_ylims = [18, 30], ax_xlims=[36.5, 38])
            sm = self.floats[i].axplot_prof(float_mids[i], prof_indices, 'seasonal', 'SA', 'pt',
                                            cbar=False, ax_ylims = [0,  30], ax_xlims=[34.5, 38])
            sb = self.floats[i].axplot_prof(float_bots[i], prof_indices, 'seasonal', 'SA', 'pt',
                                            cbar=False, ax_ylims = [1.5,  3], ax_xlims=[35.0, 35.15])
            sts.append(st)
            sms.append(sm)
            sbs.append(sb)
            
            # Set x and y labels to share some labeling between axes
            float_tops[i].set_xlabel('')
            float_mids[i].set_xlabel('')
            float_tops[i].set_ylabel('')
            float_bots[i].set_ylabel('')
            if i != 0:
                float_mids[i].set_ylabel('')
                
            # Grid
            float_tops[i].grid(True)
            float_mids[i].grid(True)
            float_bots[i].grid(True)
            
            # Set title for WMO ID
            float_tops[i].set_title(self.wmo_ids[i])
    
        # Get the average i value and remove all x labels but the middle one
        iavg = int(sum(list(range(self.n_floats)))/self.n_floats)
        for i in range(self.n_floats):
            if i != iavg:
                float_bots[i].set_xlabel('')
                
        # Step 3: Add a colorbar
        cbar = f.colorbar(sbs[-1], ax=np.ravel([float_tops, float_mids, float_bots]), 
                          ticks=np.linspace(0, 1, num=13))
        cbar.ax.set_yticklabels(['July', 'August', 'September',
                                 'October',  'November', 'December',
                                 'January', 'February', 'March',
                                 'April', 'May', 'June', 'July'])
            
        # Save and close if desired
        if saveas != None:
            f.savefig(saveas)
            plt.close(f)
        else:
            plt.show()
            
    def plot_isotherms_compare(self, day_range, saveas=None):
        """
        Mostly hardcoded plot routine for comparing isotherms between floats over 
        time. 
        """
        # Plan not to display if saving, by turning interactive mode off (or on if planning to show)
        if saveas != None:  plt.ioff()
        else:               plt.ion()
        
        # Format plot text
        rc('text', usetex=True)
        rc('font', family='serif')
        plt.style.use('dark_background')
        
        # Generate day range if needed. Format to sorted just in case.
        if day_range is None:
            day_range = self.any_match_days()
        day_range = sorted(day_range)
        
        # PREPARE AXES    
        f = plt.figure(figsize=(13,8))
        gs = gridspec.GridSpec(5, self.n_floats, figure=f)
        
        map_axis = plt.subplot(gs[ :2, : ])
        float_tops = [plt.subplot(gs[2,i]) for i in range(self.n_floats)]
        float_mids = [plt.subplot(gs[3,i]) for i in range(self.n_floats)]
        float_bots = [plt.subplot(gs[4,i]) for i in range(self.n_floats)]
        
        plt.subplots_adjust(top=0.97,bottom=0.08,left=0.055,right=0.96,hspace=0.36,wspace=0.32)
        plt.suptitle("Latest day: "   + ap.format_jd(day_range[-1]))
        
        
        # Step 1: Plot map
        self.full_map_plot(map_axis, day_range, extracolors='white')
        
        # Step 2: Plot the profiles
        sts, sms, sbs = [], [], []
        stsls, smsls, sbsls = [], [], []
        for i in range(self.n_floats):
            prof_indices = self.floats[i].profs_to_date(day_range[-1])
            st, stsl = self.floats[i].axplot_prof(float_tops[i], prof_indices, 'isotherms', 'time', 'z',
                                                  vmin=0, vmax=30, cbar=False, ax_ylims=[-500,  0    ])
            sm, smsl = self.floats[i].axplot_prof(float_mids[i], prof_indices, 'isotherms', 'time', 'z',
                                                  vmin=0, vmax=30, cbar=False, ax_ylims=[-2000, -500 ])
            sb, sbsl = self.floats[i].axplot_prof(float_bots[i], prof_indices, 'isotherms', 'time', 'z',
                                                  vmin=0, vmax=30, cbar=False, ax_ylims=[-6000, -2000])
            sts.append(st)
            sms.append(sm)
            sbs.append(sb)
            
            stsls.append(stsl)
            smsls.append(smsl)
            sbsls.append(sbsl)
            
            # Drop x ticks on top and mid axes
            float_mids[i].get_xaxis().set_ticks([])
            float_bots[i].get_xaxis().set_ticks([])
            
            # Set x and y labels to share some labeling between axes
            float_tops[i].set_xlabel('')
            float_mids[i].set_xlabel('')
            float_tops[i].set_ylabel('')
            float_bots[i].set_ylabel('')
            if i != 0:
                float_mids[i].set_ylabel('')
            
            # Set title for WMO ID
            float_tops[i].set_title(self.wmo_ids[i])
        
        # Get the average i value and remove all x labels but the middle one
        iavg = int(sum(list(range(self.n_floats)))/self.n_floats)
        for i in range(self.n_floats):
            if i != iavg:
                float_bots[i].set_xlabel('')
    
        # Step 3: Add a colorbar
        cbar = f.colorbar(sbs[-1], ax=np.ravel([float_tops, float_mids, float_bots]).flatten())
        cbar.set_label('Potential Temperature (C)', rotation=270, labelpad=15)
        cbar.add_lines(sbsls[-1])
        
        # Save and close if desired
        if saveas != None:
            f.savefig(saveas)
            plt.close(f)
        else:
            plt.show()
            
    def plot_params(self, pres_min, pres_max, param1, param2='z', 
                   delta=True, day_range=None, saveas=None, plottype='average'):
        """
        Plot a given parameter's average value over a given depth range
        
        @params
            param1     - An ArgoProfile attribute, on the gen_attrlist
            param2     - An ArgoProfile attribute, on the gen_attrlist
            pres_min  - Minimum pressure to plot from
            pres_max  - Maximum pressure to plot from
            delta     - Whether to plot absolute value or deviation from average
            day_range - Day range to plot (last day in list is top value for location/parameter plot)
            saveas    - String file path to save this plot to, as a png file, if any (otherwise call plt.show())
        """
        
        # Plan not to display if saving, by turning interactive mode off (or on if planning to show)
        if saveas != None:  plt.ioff()
        else:               plt.ion()
        
        # Generate day range if needed. Format to sorted just in case.
        if day_range is None:
            day_range = self.any_match_days()
        day_range = sorted(day_range)
        
        # Generate pressure range
        pres_range = np.arange(pres_min, pres_max, 1.0)
        
        # PREPARE AXES    
        f = plt.figure(figsize=(13,8))
        gs = gridspec.GridSpec(6, self.n_floats, figure=f)
        map_axes = [plt.subplot(gs[ 0:2 , i]) for i in range(self.n_floats)]
        par_axes = [plt.subplot(gs[ 2:4 , i]) for i in range(self.n_floats)]
        day_axis = plt.subplot(gs[4:6, :])
        
        # Add axis labels and format shape to place nicer when saving without adjustments. 
        for par_axis in par_axes:
            par_axis.set_xlabel(ap.attr_to_name(param1))
            par_axis.set_ylabel(ap.attr_to_name(param2))
        day_axis.set_xlabel("Date")
        day_axis.set_ylabel(str(param1) + " " + plottype)
        plt.tight_layout()
        plt.subplots_adjust(top=0.94, bottom=0.203, left=0.081, right=0.98, hspace=0.751, wspace=0.481)
        
        
        # AXES : map_axes : Float location on map with SLA
        
        # For each argo float
        for argofloat, map_axis, wmo_id in zip(self.floats, map_axes, self.wmo_ids):
            
            # a. Draw the map
            map = argofloat.aviso_map_plot(day_range[-1], map_axis)       
            
            # b. Turn float interpolated lons and lats for this day into map coordinates
            x, y = map(argofloat.lon_interp(day_range), argofloat.lat_interp(day_range))
            
            # c. Plot, emphasizing the latest position as a green dot
            map.plot(x, y, color='black', zorder=2)
            map.scatter(x[-1], y[-1], color='green', s=5, zorder=3)
            
            # d. Label axis
            map_axis.set_title(str(wmo_id))
        
        
        # AXES : par_axes : Parameter vs. Depth for each profile
        
        # 1. Prepare a set of colors to emphasize most recent profile info and grey out old information
        cmap = cm.get_cmap('binary')
        norm = matplotlib.colors.PowerNorm(vmin=1, vmax=5, gamma=3.0, clip=True)
        colors = [cmap(0.05+0.95*norm(i+1-len(day_range)+5)) for i in range(len(day_range))]
        
        # 2. Go through the days and...
        avg1 = [[] for d in day_range]  #prep to get averages w/ (day, float) indexing
        avg2 = [[] for d in day_range]  #prep to get averages w/ (day, float) indexing
        for i_d in range(len(day_range)):
            
            # a. Go through each synthetic profile and generate the desired parameters' values
            param1_values = []
            param2_values = []
            for synthprof in self.synth_profiles[day_range[i_d]]:
                param1_values.append(getattr(synthprof, param1+'_interp')(pres_range))
                param2_values.append(getattr(synthprof, param2+'_interp')(pres_range))
            
            # b. And iterate through the floats...
            for i_f in range(self.n_floats):
                # Either plot the values or the delta from average values, and add the average to the list
                if delta:
                    
                    #1. Get the mean values of the parameters first
                    param1_means = getattr(self.floats[i_f].prof_avg, param1+'_interp')(pres_range)
                    param2_means = getattr(self.floats[i_f].prof_avg, param2+'_interp')(pres_range)
                    
                    #2. Get the differential between this day and the average over time
                    delta1_values = param1_values[i_f] - param1_means
                    delta2_values = param2_values[i_f] - param2_means
                    
                    #3. Append the average to the list of averages
                    avg1[i_d].append(np.nanmean(delta1_values))
                    avg2[i_d].append(np.nanmean(delta2_values))
                    
                    #4. Plot on the corresponding parameter axis
                    par_axes[i_f].plot(delta1_values[i_f], param2_values[i_f], color=colors[i_d])
                    
                else:
                    
                    #1. Add the not-differential average to the list of averages
                    avg1[i_d].append(np.nanmean(param1_values[i_f]))
                    avg2[i_d].append(np.nanmean(param2_values[i_f]))
                    
                    #2. Plot on corresponding parameter axis
                    par_axes[i_f].plot(param1_values[i_f], param2_values[i_f], color=colors[i_d])
    
        # 3.  Grid the par_axis plots
        for par_axis in par_axes:
            par_axis.grid(True)
        
        # 4. Plot the average as a verticle line on the axis
        for par_axis, a in zip(par_axes, avg1[-1]):
            par_axis.axvline(a, color='red')
            
        
        # AXIS : day_axis : Parameter Average vs. Time
        
        # 1. Change average indexing from (day, float) to (float, day)
        avg1 = np.array(avg1).T
        
        # 2. Plot the average over time for each float
        plt.sca(day_axis)
        for i in range(self.n_floats): 
            day_axis.plot(day_range, avg1[i], label=self.wmo_ids[i], linewidth=2)
        
        # 3. Plot the average between floats over time
        day_axis.plot(day_range, np.mean(avg1, axis=0), label='Average', linewidth=3, color='black')
        
        # 4. Add the grid and legend
        day_axis.grid(True)
        day_axis.legend(loc='lower right')
        
        # 5. Format date on x axis tick labels
        xticks = day_axis.get_xticks() 
        form = [ap.format_jd(xt) for xt in xticks]     # Convert into formatted dates
        day_axis.set_xticklabels(form, rotation=90)
        
        
        # Save and close if desired
        if saveas != None:
            f.savefig(saveas)
            plt.close(f)
        else:
            plt.show()
    
    def prep_interp_profs(self, day_range):
        """
        Interpolate the ArgoFloat's Profile objects contained in this 
        FloatComparer object onto the given day range, creating a set of 
        synthetic profiles for each day, interpolated from the real profiles. 
        """
        for day in sorted(day_range):
            if day not in self.synth_profiles:
                synths = [argofloat.day_profile(day) for argofloat in self.floats]
                
                for synthprof in synths:
                    synthprof.infer_values()
                    synthprof.make_interps()
                    
                self.synth_profiles.update({day:synths})
    
    def comps_over_time(self, savedir='/home/cassandra/docs/argo/movies/comparer/comps/',
                        profs=True, iso=True):
        """
        
        """
        af.ensure_dir(savedir)
        day_range = self.any_match_days()
        day_range = np.linspace(min(day_range), max(day_range),
                                26*(max(day_range) - min(day_range))/365, dtype=int)
        if iso and not profs:
            af.ensure_dir(savedir+'isotherms/')
            for i in range(1, len(day_range)):
                print('Plotting index', i, 'of', len(day_range))
                self.plot_isotherms_compare(day_range[:i+1], saveas=savedir+'isotherms/f{:0>3d}.png'.format(i))
        elif profs and not iso:
            af.ensure_dir(savedir+'profiles/')
            for i in range(1, len(day_range)):
                print('Plotting index', i, 'of', len(day_range))
                self.plot_profiles_compare(day_range[:i+1], saveas=savedir+'profiles/f{:0>3d}.png'.format(i))
        elif profs and iso:
            af.ensure_dir(savedir+'profiles/')
            af.ensure_dir(savedir+'isotherms/')
            for i in range(1, len(day_range)):
                print('Plotting index', i, 'of', len(day_range))
                self.plot_profiles_compare(day_range[:i+1], saveas=savedir+'profiles/f{:0>3d}.png'.format(i))
                self.plot_isotherms_compare(day_range[:i+1], saveas=savedir+'isotherms/f{:0>3d}.png'.format(i))       
            
    def plot_over_time(self, pres_min, pres_max, param1, param2='z', delta=False,
                       savedir='/home/cassandra/docs/argo/movies/comparer/'):
        """
        
        """
        if delta:
            savedir = savedir + 'delta/'
        savedir = savedir + param1 + "/" + param2 + "/" + str(pres_min) + "/" + str(pres_max) + "/"
        af.ensure_dir(savedir)
        day_range = self.any_match_days()
        day_range = np.linspace(min(day_range), max(day_range),
                                26*(max(day_range) - min(day_range))/365, dtype=int)
        self.prep_interp_profs(day_range)
        for i in range(1, len(day_range)):
            print('Plotting index', i, 'of', len(day_range))
            self.plot_params(pres_min, pres_max, param1, param2 = param2, 
                             delta=delta, day_range = day_range[:i+1],
                             saveas = savedir+'f{:0>3d}.png'.format(i))
                    
    def bathymetry_over_time(self, savedir='/home/cassandra/docs/argo/movies/bathymetry/'):
        """
        
        """
        af.ensure_dir(savedir)
        day_range = self.any_match_days()
        day_range = np.linspace(min(day_range), max(day_range),
                                26*(max(day_range) - min(day_range))/365, dtype=int)
        for i in range(1, len(day_range)):
            print('Plotting index', i, 'of', len(day_range))
            self.plot_bathymetry_position(day_range = day_range[:i+1],
                                          saveas = savedir+'f{:0>3d}.png'.format(i))
                           
        
            
            
def main():
    wmo_ids = [4902326, 4902324, 4902323, 4902322, 4902321]
    wmo_ids = [4902326, 4902324, 4902323, 4902322, 4902321]
    
    comparer = FloatComparer(wmo_ids, 
                             argo_dir="/data/deep_argo_data/nc/",
                             aviso_dir = '/data/aviso_data/nrt/weekly/')
    
    #comparer.plot_isotherms_together()
    #comparer.bathymetry_over_time()
    comparer.plot_map_only()
    
    #comparer.comps_over_time()
    #comparer.plot_over_time( 0, 5500, 'pt', param2='SA')
    
    #comparer.plot_over_time( 750, 1250, 'pt')
    #comparer.plot_over_time(1750, 2250, 'pt')
    
    #comparer.plot_over_time('SA', 4500, 5500)
    #comparer.plot_over_time('pt', 4500, 5500)

if __name__ == '__main__':
    main()