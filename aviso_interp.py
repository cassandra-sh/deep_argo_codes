#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 11:46:55 2018

@author: cassandra
"""

import glob
import xarray   as xr
import datetime as dt
import pandas   as pd
import numpy    as np
import sys
import time
import gc
import pickle
import scipy.interpolate

def nc_interp(file, col, **kwargs):
    """
    Open an nc file, interpolate for a subset given a column and some kwargs.
    """
    with xr.open_dataset(file) as xf:
        return xf.interp(**kwargs)[col]

def datetime_to_argo_juld(date):
    """
    Take a numpy datetime64 object and calculate the argo julian day style
    date for the object. 
    
    Argo julian day is the number of days since 1950-01-01. Conversion is thus
    the actual julian day minus the julian day of 1950-01-01 as noted below:
    
    astropy.time.Time('1950-01-01').jd = 2433282.5
    """
    #print('avisio_interp.datetime_to_argo_juld', date, pd.to_datetime(date).to_julian_date(), 
    #      pd.to_datetime(date).to_julian_date() - 2433282.5 )
    return pd.to_datetime(date).to_julian_date() - 2433282.5 #minus reference time for argo juld

def norm_lon(lon):
    """
    
    """
    if lon > 0 and lon < 360:
        return lon
    elif lon < 0:
        while lon < 0:
            lon = lon + 360
        return lon
    else:
        while lon > 360:
            lon = lon - 360
        return lon

def aviso_read(file, nv=0, box=None):
    """
    Read an aviso nc file and return the following for constructing an ND 
    interpolator
    
    -matrix of sla values
    -longitudes
    -latitudes
    -argo-style julian day
    
    
            box       - [lower lon, upper lon, lower lat, upper lat]
    
    """
    with xr.open_dataset(file) as xf:
        
        if box != None:
            xf = xf.where(xf.lon > norm_lon(box[0]), drop=True)
            xf = xf.where(xf.lon < norm_lon(box[1]), drop=True)
            xf = xf.where(xf.lat > box[2], drop=True)
            xf = xf.where(xf.lat < box[3], drop=True)
        
        lat = xf.lat.values
        lon = xf.lon.values
        day = datetime_to_argo_juld(xf['time'].values[0])
        mat = xf.sel(nv=nv)['sla'].values[0]
    
    gc.collect()
    return mat, lon, lat, day
    
def dims_to_points(*dims):
    """
    
    """
    points = []
    for mm in np.meshgrid(*dims):
        points.append(mm.flatten())
    return np.array(points).T

def dropnan(points, vals):
    good = np.where(np.logical_not(np.isnan(vals)))[0]
    return [points[i] for i in good], [vals[i] for i in good]

class AvisoInterpolator:
    """
    
    """
    
    def __init__(self, **kwargs):
        """
        
        @params **kwargs
            aviso_dir - 
            verbose   - 
            load      - 
            save      - 
            irregular - 
            limit     -
            box       - [lower lon, upper lon, lower lat, upper lat]
            units     - 'm' or 'cm' - the units of the nc files used
                        interpolated values will ALWAYS be in meters.
        """
        verbose = kwargs.get('verbose', True)
        load = kwargs.get('load', False)
        save = kwargs.get('save', False)
        limit = kwargs.get('limit', 6)
        
        self.aviso_dir = kwargs.get('aviso_dir', '/data/aviso_data/monthly_mean/')
        self.aviso_files = glob.glob(self.aviso_dir+'*.nc')
        self.aviso_files = sorted(self.aviso_files)[-1*limit:]
        
        #
        # Load an interpolator?
        #
        if load:
            obj = open(self.aviso_dir+"interp.pickle", 'rb')
            self.interp = pickle.load(obj)
        
        #
        # Actually make an interpolator by loading out the Aviso data
        #
        else:
            mats, self.lons, self.lats, self.days = [], [], [], []
            for f in self.aviso_files:
                mat, lon, lat, day = aviso_read(f, box=kwargs.get('box', None))
                mats.append(mat)
                self.days.append(day)
                self.lons = lon
                self.lats = lat
            
            mats = np.array([x for _,x in sorted(zip(self.days,mats))])
            if kwargs.get('units', 'm') == 'cm':
                mats = mats/100.0  #CONVERT FROM CM TO M
                
            self.days = np.array(sorted(self.days))
            n_nan =  len(np.where(np.isnan(mats.flatten()))[0])
            
            #
            # Generate the interpolator object in one of two flavors
            #
            if verbose:
                print("About to make interpolator. Starting the clock...")
                start_time = time.time()
                sys.stdout.flush()
                
            irregular = kwargs.get("irregular", 'guess')
            if irregular == 'guess':
                if n_nan > 0:
                    irregular = True
                else:
                    irregular = False
            self.irregular = irregular
            
            if irregular:
                points = dims_to_points(self.days, self.lats, self.lons)
                values = mats.flatten()
                points, values = dropnan(points, values)
                self.interp = scipy.interpolate.LinearNDInterpolator(points, values)
            else:
                dims = (self.days, self.lats, self.lons)
                self.interp = scipy.interpolate.RegularGridInterpolator(dims, mats, 
                                                                        fill_value=np.nan,
                                                                        bounds_error=False)
            
            if verbose:
                print("Made the interpolator! Time elapsed in seconds is " + str(int(time.time() - start_time)))
                print("Number of nans in mats is " + str(len(np.where(np.isnan(mats.flatten()))[0])))
                print("AvisoInterpolator initialized with day range", min(self.days), "to", max(self.days))
                sys.stdout.flush()
            
            #
            # Store the max and min of the interpolator
            #
            self.max = np.nanmax(mats)
            self.min = np.nanmin(mats)
            
            mats = None
            gc.collect()
        
        #
        # Save an interpolator?
        #
        if save:
            obj = open(self.aviso_dir+"interp.pickle", 'wb')
            pickle.dump(self.interp, obj)
        
    def interpolate(self, *args):
        if self.irregular:
            return self.interp(*args)
        else:
            return self.interp(tuple(args))
        
    def interp_sla(self, date, lon, lat):
        """
        
        """
        ins = np.array([date, lon, lat]).T
        return self.interp(ins)
    
    def close(self):
        """
        
        """
        self.interp = None
        gc.collect()

def main():
    argo_float = xr.open_dataset('/data/argo_data/nc/D6901509_069.nc')
    date = datetime_to_argo_juld(argo_float['JULD'].values[0])
    lon = argo_float['LONGITUDE'].values[0]
    lat = argo_float['LATITUDE'].values[0]
    
    AI = AvisoInterpolator()
    print(date, lon, lat, AI.interp_sla(date, lon, lat))
    AI.close()
    
if __name__ == "__main__":
    main()
    