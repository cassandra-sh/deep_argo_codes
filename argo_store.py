#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 13:49:03 2018

@author: cassandra
"""

import pandas as pd
import re
import numpy as np
import urllib
import xarray as xr
import sys
import os
import glob
import shutil
import multiprocessing as mp
import which_ocean




class ArgoStore:
    """
    
    """
    
    def __init__(self, **kwargs):
        """
        
        """
        self.index_file = kwargs.get('index_file', "/home/cassandra/docs/argo/small_data/ar_index_global_prof.txt")
        self.index_file_url = kwargs.get('index_file_url', 'http://www.usgodae.org/ftp/outgoing/argo/ar_index_global_prof.txt')
        
        renew_index = kwargs.get('renew_index', False)
        if renew_index:
            self.renew_index()
        else:
            self.index_df = pd.read_csv(self.index_file, comment="#", header=0)
        
        self.data_dir   = kwargs.get('data_dir', "/data/")
        self.ftp_url    = kwargs.get('ftp_url', 'http://www.usgodae.org/ftp/outgoing/argo/dac/')
        
        self.od = which_ocean.OceanDecider()
        
        pass
    
    def renew_index(self, **kwargs):
        """
        
        """
        os.remove(self.index_file)
        download_file(self.index_file_url, self.index_file)
        self.index_df = pd.read_csv(self.index_file_url, comment='#', header=0)
    
    def download_argo(self, **kwargs):
        """
        Download the argo data associated with the **kwargs parameters given.
        
        @**kwargs            
            name         - default - description
            ------------------------------------------------------------------
            check_done   - True    - Whether or not to check that the file to
                                     be downloaded is already in the data dir.
            start        - 2014    - Starting year to get data in
            stop         - np.inf  - Stop year to get data in
            natlantic    - True    - Whether or not to only take profiles from
                                     the north atlantic 
            D_only       - True    - Whether or not to take display only files
            deep         - False   - Download argo deep
            multiprocess - True    -
            n_threads    - 4       -
        """
        #Check if will download argo regular or deep
        #Use different directories for both
        deep = kwargs.get('deep', False)
        if deep:
            inst_data_dir = self.data_dir+'deep_argo_data/'
        else:
            inst_data_dir = self.data_dir+'argo_data/'
        
        
        
        #Now using the index dataframe, figure out what files meet the download conditons
        conditions = []
        
        
        #Already downloaded check
        check_done = kwargs.get('check_done', True) 
        if check_done:
            already_done = set(glob.glob(inst_data_dir+'nc/*.nc'))
            index_file_names = [(inst_data_dir+'nc/'+f.split('/')[-1]) for f in self.index_df['file']]
            conditions.append(np.array([(f not in already_done) for f in index_file_names]))


        #Date requirements 
        start = kwargs.get('start', 2014)
        stop  = kwargs.get('stop', np.inf) 
        start_date = start*1E+10
        stop_date  = stop*1E+10      
    
        conditions.append(np.greater(self.index_df['date'].values, start_date))
        conditions.append(np.less(self.index_df['date'].values, stop_date))
        
        
        #Location requirements
        natlantic  = kwargs.get('natlantic', True)
        if natlantic:
            lons = self.index_df['longitude'].values
            lats = self.index_df['latitude'].values
            conditions.append(self.od.in_natlantic(lons, lats))
        
        
        #Display ready requirements
        D_only = kwargs.get('D_only', True)
        if D_only:
            r = re.compile('.*/.*/.*/D.*.nc')
            conditions.append(np.array([(r.match(f) != None) for f in self.index_df['file']]))
        
        
        #Deep Argo requirements (profiler type = 862)
        if deep:
            conditions.append(np.array(self.index_df['profiler_type'].values) == 862)
            
            
        #Put them all together into a list and get the list of files to download
        dl_needed = and_them(*conditions)
        url_list = [self.ftp_url+self.index_df['file'][i] for i in np.where(dl_needed)[0]] 
        target_list = [inst_data_dir+'nc/'+self.index_df['file'][i].split('/')[-1] for i in np.where(dl_needed)[0]] 
    
        #Download everything, multiprocess if preferred
        multiprocess = kwargs.get('multiprocess', True)
        n_threads = kwargs.get('n_threads', 4)
        if multiprocess:
            lo_divs = np.linspace(0, len(url_list), n_threads+1, dtype=int)[:-1]
            up_divs = np.linspace(0, len(url_list), n_threads+1, dtype=int)[1:]
            
            procs = []
            for i in range(n_threads):
                arg = (url_list[lo_divs[i]:up_divs[i]], target_list[lo_divs[i]:up_divs[i]],)
                procs.append(mp.Process(target=download_list, args=arg))
            for proc in procs:
                proc.daemon = True
                proc.start()
            for proc in procs:
                proc.join()   
        else:
            download_list(url_list, target_list)
            
    def nc_to_hdf(self, **kwargs):
        """
        Convert existing nc files to hdf files
        
        @params **kwargs
            deep         -
            check_done   -
            multiprocess -
            n_threads    -
        """
        
        #Check if will download argo regular or deep
        #Use different directories for both
        deep = kwargs.get('deep', False)
        if deep:
            inst_data_dir = self.data_dir+'deep_argo_data/'
        else:
            inst_data_dir = self.data_dir+'argo_data/'
        
        # Get lists of files already downloaded and converted
        nc_files = glob.glob(inst_data_dir+'nc/*.nc')
        targ_files = [inst_data_dir+'hdf/'+f.split('/')[-1]+'.hdf' for f in nc_files]
        hdf_files = set(glob.glob(inst_data_dir+'hdf/*.hdf'))
        
        #Already converted check
        check_done = kwargs.get('check_done', True)
        
        #If yes, only plan to convert files that have not yet been converted
        if check_done:
            not_converted = np.where([(file not in hdf_files) for file in targ_files])[0]
            source_list = [nc_files[i] for i in not_converted]
            target_list = [targ_files[i] for i in not_converted]
        
        #If no, delete all files to later replace them with newly converted files
        else:
            for file in targ_files:
                if file in hdf_files:
                    os.remove(file)
            source_list = nc_files
            target_list = targ_files
            
        #Convert everything, multiprocess if preferred
        multiprocess = kwargs.get('multiprocess', True)
        n_threads = kwargs.get('n_threads', 4)
        if multiprocess:
            lo_divs = np.linspace(0, len(target_list), n_threads+1, dtype=int)[:-1]
            up_divs = np.linspace(0, len(target_list), n_threads+1, dtype=int)[1:]
            
            procs = []
            for i in range(n_threads):
                arg = (source_list[lo_divs[i]:up_divs[i]], target_list[lo_divs[i]:up_divs[i]],)
                procs.append(mp.Process(target=download_list, args=arg)) 
            for proc in procs:
                proc.daemon = True
                proc.start()
            for proc in procs:
                proc.join()   
        else:
            download_list(source_list, target_list)       
                



def and_them(*args):
    """
    numpy logical and the arguments given. Must all be the same length
    """
    out = np.ones(len(args[0]), dtype=bool)
    for arg in args:
        out = np.logical_and(arg, out)
    return out



def download_list(url_list, target_list):
    """
    Download a list of files and read out a simple progress update every 500 files
    """
    n_files = len(url_list)
    progress = range(len(url_list))
    for url, target, prog in zip(url_list, target_list, progress):
        download_file(url, target)
        
        #Simple progress report
        if prog % 1000 == 0:
            print(int(100*prog/n_files), '% downloaded...')
            sys.stdout.flush()

def download_file(url, target):
    """
    Download the given file from some url to a target file on disk
    """
    with urllib.request.urlopen(url) as response:
        with open(target, 'wb') as target_file:
            shutil.copyfileobj(response, target_file)

def nc_to_hdf_list(source_list, target_list):
    """
    Turn a list of files into HDFs
    """
    n_files = len(source_list)
    progress = range(len(source_list))
    for url, target, prog in zip(source_list, target_list, progress):
        nc_to_hdf(url, target)
        
        #Simple progress report
        if prog % 1000 == 0:
            if prog > 0:
                print(int(100*prog/n_files), '% converted...')
                sys.stdout.flush()

def nc_to_hdf(source, target):
    """
    Turn the given xarray readable file (designed for NetCDF 3 Classic) into
    an HDF. Resulting HDF will often take up much more space if multiple
    coordinates are used for the NetCDF file. 
    """
    with xr.open_dataset(source, decode_times=False) as source_file:
        df = source_file.to_dataframe()
        df.to_hdf(target, 'p')


def main():
    AS = ArgoStore()
    AS.download_argo()

if __name__ == "__main__":
    main()
    
