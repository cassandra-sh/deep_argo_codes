#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 11:20:02 2018

@author: cassandra

Script to take out the .mat files provided by Matt Mazloff and save weekly
averages as NC files that aviso_interp.py can read


"""

import scipy.io
import glob
import gc
import sys
import xarray as xr
import datetime as dt
import numpy as np
import h5py
import pandas as pd


def datetime_to_argo_juld(date):
    """
    Take a numpy datetime64 object and calculate the argo julian day style
    date for the object. 
    
    Argo julian day is the number of days since 1950-01-01. Conversion is thus
    the actual julian day minus the julian day of 1950-01-01
    """
    return pd.to_datetime(date).to_julian_date() - 2433282.5

def matlab2datetime(matlab_datenum):
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=int(matlab_datenum)%1) - dt.timedelta(days = 366)
    return day + dayfrac

def main(combine = False, average = True):
    
    
    
    if combine:
        
        
        mat_dir = '/data/aviso_data/nrt/'
        mat_files = glob.glob(mat_dir+"*.mat")
        
        dct1 = h5py.File(mat_files[1], 'r')
        dct2 = scipy.io.loadmat(mat_files[0])
        
        times = np.append(dct1['tsave'][0], dct2['tsave'][0]).flatten().tolist()
        times = [matlab2datetime(time) for time in times]
        lons  = np.array(dct1['xsave']).flatten()
        lats  = np.array(dct1['ysave']).flatten()
        sla1 = np.array(dct1['dsave'])
        sla2 = np.array(dct2['dsave'])
        
        dct1, dct2 = None, None
        gc.collect()
        
        sla   = np.append(sla1, sla2.T, axis=2)
        
        sla1, sla2 = None, None
        gc.collect()
        
        da = xr.DataArray(sla, dims=['longitude', 'latitude', 'time'],
                          coords={'longitude':lons, 'latitude':lats, 'time':times})
        
        sla, lons, lats, times = None, None, None, None
        gc.collect()
        
        xf = xr.Dataset({'sla':da})
        
        da = None
        gc.collect()
        
        xf.to_netcdf(mat_dir+'mat_dat.nc')
        
        xf = None
        gc.collect()
        
    if average:
        mat_dir = '/data/aviso_data/nrt/'
        xf = xr.open_dataset(mat_dir+'mat_dat.nc')
        
        save_dir = '/data/aviso_data/nrt/weekly/'
        
        lons = xf.longitude.values
        lats = xf.latitude.values
        
        times = np.flip(xf.time.values, 0)
        
        for i, time in enumerate(times):
            jd = int(datetime_to_argo_juld(time))
            if jd % 7 != 0:
                pass
            else:
                form_time = str(pd.to_datetime(time).strftime('%Y_%m_%d'))
                print('Getting average for week ' + form_time + ' which is ' +
                      str(int(i/7)) + ' of ' + str(int(len(times)/7)))
                sys.stdout.flush()
                
                sla = xf.where(xf.time >= time - np.timedelta64(3, 'D'), drop=True)
                sla = sla.where(xf.time <= time + np.timedelta64(3, 'D'), drop=True)
                sla = [[np.nanmean(sla['sla'], axis=2).T]]
                
                da = xr.DataArray(sla, dims=['time', 'nv', 'lat', 'lon'], coords={'lon' : lons,
                                                                                  'lat' : lats,
                                                                                  'nv'  : [0],
                                                                                  'time': [time]})
                new_xf = xr.Dataset({'sla':da})
                new_xf.to_netcdf(save_dir+"aviso_week_avg_"+form_time+'.nc')
                
                sla, new_xf, da, form_time = None, None, None, None
                gc.collect()

if __name__ == "__main__":
    main()