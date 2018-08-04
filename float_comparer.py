#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 14:15:08 2018

@author: cassandra
"""
import numpy        as np
import argo_float   as af
import argo_profile as ap

import matplotlib
import matplotlib.gridspec as     gridspec
import matplotlib.pyplot   as     plt
from matplotlib.pyplot     import cm

class FloatComparer:
    """
    Compare some ArgoFloat objects at different depths
    """
    
    def __init__(self, wmo_ids, **kwargs):
        """
        Initialize the FloatComparer with a list of WMO_IDs and **kwargs to pass
        to the ArgoFloat initializer. 
        
        """
        self.wmo_ids = wmo_ids
        self.floats = [af.ArgoFloat(wmo_id, **kwargs) for wmo_id in wmo_ids]
        self.n_floats = len(self.floats)
    
    def any_match_days(self):
        """
        Return all days, sorted, which have aviso data and at least one argo float's data
        """
        return sorted(list(set().union(*[argofloat.match_days() for argofloat in self.floats])))
    
    def plot_depth(self, param,  pres_min, pres_max,
                   delta=True, day_range=None, saveas=None):
        """
        Plot a given parameter's average value over a given depth range
        
        @params
            param     - An ArgoProfile attribute, on the gen_attrlist
            pres_min  - Minimum pressure to plot from
            pres_max  - Maximum pressure to plot from
            delta     - Whether to plot absolute value or deviation from average
            day_range - Day range to plot (last day in list is top value for location/parameter plot)
            saveas    - File to save this plot as, if any (otherwise call plt.show())
        """
        
        # Plan not to display if saving
        if saveas != None:  plt.ioff()
        else:               plt.ion()
        
        if day_range == None:
            day_range = self.any_match_days()
        pres_range = np.arange(pres_min, pres_max, 1.0)
        day_range = sorted(day_range)
        
        # PREPARE AXES    
        f = plt.figure(figsize=(13,8))
        gs = gridspec.GridSpec(6, self.n_floats, figure=f)
        map_axes = [plt.subplot(gs[ 0:2 , i]) for i in range(self.n_floats)]
        par_axes = [plt.subplot(gs[ 2:4 , i]) for i in range(self.n_floats)]
        day_axis = plt.subplot(gs[4:6, :])
        
        for par_axis in par_axes:
            par_axis.set_xlabel(ap.attr_to_name(param))
            par_axis.set_ylabel('Depth (m)')
        day_axis.set_xlabel("Date")
        day_axis.set_ylabel(str(param) + " average")
        plt.tight_layout()
        plt.subplots_adjust(top=0.94,
                            bottom=0.203,
                            left=0.081,
                            right=0.98,
                            hspace=0.751,
                            wspace=0.481)
        
        # AXES : map_axes : Float location on map with SLA
        for argofloat, map_axis, wmo_id in zip(self.floats, map_axes, self.wmo_ids):
            map = argofloat.aviso_map_plot(day_range[-1], map_axis)            
            x, y = map(argofloat.lon_interp(day_range), argofloat.lat_interp(day_range))
            map.plot(x, y, color='black', zorder=2)
            map.scatter(x[-1], y[-1], color='green', s=5, zorder=3)
            map_axis.set_title(str(wmo_id))
        
        
        # AXES : par_axes : Parameter vs. Depth for each profile
        
        # 1. Prepare a set of colors to emphasize most recent profile info and grey out old information
        cmap = cm.get_cmap('binary')
        norm = matplotlib.colors.PowerNorm(vmin=1, vmax=5, gamma=3.0, clip=True)
        colors = [cmap(0.05+0.95*norm(i+1-len(day_range)+5)) for i in range(len(day_range))]
        
        # 2. Go through the days and...
        avg = [[] for d in day_range]  #prep to get averages w/ (day, float) indexing
        for i, day in enumerate(day_range):
            
            # a. Make synthetic profiles for this day from each float
            synth_profiles = [argofloat.day_profile(day) for argofloat in self.floats]
            
            # b. And iterate through the snythetic profiles, axes, and argo floats...
            for synthprof, par_axis, argofloat in zip(synth_profiles, par_axes, self.floats):
                
                # 1. Prepare the synthetic profiles for interpolating...
                synthprof.infer_values()
                synthprof.make_interps()
                
                # 2. Interpolate to get the desired parameters and depth
                param_values = getattr(synthprof, param+'_interp')(pres_range)
                depth_values = synthprof.z_interp(pres_range)
                
                # 3. Either plot the values or the delta from average values, and add the average to the list
                if delta:
                    param_means = getattr(argofloat.prof_avg, param+'_interp')(pres_range)
                    delta_values = param_values - param_means
                    avg[i].append(np.nanmean(delta_values))
                    par_axis.plot(delta_values, depth_values, color=colors[i])
                else:
                    avg[i].append(np.nanmean(param_values))
                    par_axis.plot(param_values, depth_values, param_values=colors[i])
    
        # Grid the par_axis plots
        for par_axis in par_axes:
            par_axis.grid(True)
        
        # Plot the average as a verticle line on the axis
        for par_axis, a in zip(par_axes, avg[-1]):
            par_axis.axvline(a, color='red')
            
        
        # AXIS : day_axis : Parameter Average vs. Time
        
        # Change average indexing from (day, float) to (float, day)
        avg = np.array(avg).T
        
        # Plot
        plt.sca(day_axis)
        for i in range(self.n_floats): 
            day_axis.plot(day_range, avg[i], label=self.wmo_ids[i], linewidth=2)
        day_axis.grid(True)
        day_axis.legend(loc='lower right')
        
        # Format date on x axis tick labels
        xticks = day_axis.get_xticks() 
        form = [ap.format_jd(xt) for xt in xticks]     # Convert into formatted dates
        day_axis.set_xticklabels(form, rotation=90)
        
        # Save and close if desired
        if saveas != None:
            f.savefig(saveas)
            plt.close(f)
        else:
            plt.show()
            
    def plot_over_time(self, param, pres_min, pres_max, delta=True,
                       savedir='/home/cassandra/docs/argo/movies/comparer/'):
        savedir = savedir + param + "/" + str(pres_min) + "/" + str(pres_max) + "/"
        af.ensure_dir(savedir)
        day_range = self.any_match_days()
        day_range = np.linspace(min(day_range), max(day_range),
                                64*(max(day_range) - min(day_range))/365, dtype=int)
        for i in range(1, len(day_range)):
            print('Plotting index', i, 'of', len(day_range))
            self.plot_depth(param, pres_min, pres_max, delta=delta,
                            day_range = day_range[:i+1],
                            saveas    = savedir+'f{:0>3d}.png'.format(i))
                           
        
            
            
def main():
    wmo_ids = [4902321, 4902322, 4902323, 4902324, 4902326]
    
    comparer = FloatComparer(wmo_ids, 
                             argo_dir="/data/deep_argo_data/nc/",
                             aviso_dir = '/data/aviso_data/nrt/weekly/')
    
    comparer.plot_over_time('pt', 750, 1250)
    comparer.plot_over_time('pt', 1750, 2250)
    comparer.plot_over_time('pt', 4500, 5500)
    
    comparer.plot_over_time('sa', 4500, 5500)

if __name__ == '__main__':
    main()