#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 10:29:36 2018

@author: cassandra
"""

import fiona
import pandas as pd

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot      as plt
import matplotlib.collections as mpl_collections
import matplotlib.patches     as mpl_patches

import numpy as np
import shapely.geometry

NATLANTIC_OS = ['Caribbean Sea', 'North Atlantic Ocean', 'Gulf of Mexico',
                'Straits of Florida', 'Labrador Sea']
OCEAN_POLY= '/home/cassandra/docs/argo/small_data/ne10m_labeled/ne_10m_geography_marine_polys.shp'

class OceanDecider:
    """
    
    """
    
    def __init__(self, **kwargs):
        """
        Create OceanDecider object, load up the ocean polygons
        
        @params 
            ocean_poly
                .shp file which contains the polygon information associated '
                with the global oceans.
            natlantic_os
                the names in the ocean_poly file associated with the north
                atlantic ocean, to match to 
                ocean_poly_file['parameters']['name']
        """
        ocean_poly = kwargs.get('ocean_poly', OCEAN_POLY)
        self.c = fiona.open(ocean_poly)
        
        self.natlantic_os = kwargs.get('natlantic_os', NATLANTIC_OS)
        
        
        #
        # Get out the polygons associated with the north Atlantic
        #
        natlantic_os_polys = []
        for cc in self.c:
            if cc['properties']['name'] in self.natlantic_os:
                natlantic_os_polys.append(cc['geometry'])
        self.natlantic_ocean_polys = [shapely.geometry.shape(p) for p in natlantic_os_polys]
        
    
    def in_natlantic(self, lon, lat):
        """
        Return true whether or not location(s) are in north atlantic
        
        Probably a rather inefficient way to do it
        
        @params
            lon, lat
            floats or iterable of floats for longitude and latitude
        """
        
        #
        # Case of iterable of points passed
        #
        try:
            #
            # Comprise a list of shapely.geometry.shape objects
            #
            check_points = []
            for lo, la in zip(lon, lat):
                dct = {'coordinates':[lo, la],'type':'point'}
                check_points.append(shapely.geometry.shape(dct))
            
            #
            # Check if each point is in any of the oceans
            #
            points_good = []
            for point in check_points:
                in_oceans = [point.within(ocean) for ocean in self.natlantic_ocean_polys]
                if True in in_oceans:
                    points_good.append(True)
                else:
                    points_good.append(False)
            
            return np.array(points_good, dtype=bool)
        
        #
        # Case of only 1 point passed
        #
        except TypeError:
            #
            # Same but for 1 point
            #
            dct = {'coordinates':[lon, lat],'type':'point'}
            point = shapely.geometry.shape(dct)
            in_oceans = [point.within(ocean) for ocean in self.natlantic_ocean_polys]
            if True in in_oceans:
                return True
            else:
                return False
        
        
def main():

    #
    # Get the list of argo float profiles
    #
    df = pd.read_csv("/home/cassandra/docs/argo/small_data/ar_index_global_prof.txt",
                     header=0, comment="#")
                     
    #
    # Take only floats in 2018
    #
    df = df.query('date >= 20180000000000')

    #
    # Make a decider
    #
    od = OceanDecider()

    #
    # Get the argo floats only in the atlantic
    #
    longitudes = df['longitude']
    latitudes = df['latitude']
    in_natlantic = np.where(od.in_natlantic(longitudes, latitudes))[0]
    
    #
    # Prep a basemap to plot on
    # 
    fig     = plt.figure()
    ax      = fig.add_subplot(111)
    map = Basemap()
    map.fillcontinents(color='#ddaa66',lake_color='aqua')
    map.drawcoastlines()
    map.readshapefile(OCEAN_POLY[:-4], 'oceans', drawbounds = False)
    
    
    #
    # Plot the atlantic ocean regions to be used
    #
    patches = []
    for info, shape in zip(map.oceans_info, map.oceans):
        if info['name'] in NATLANTIC_OS:
            patches.append( mpl_patches.Polygon(np.array(shape), True) )
    
    ax.add_collection(mpl_collections.PatchCollection(patches, facecolor= 'm',
                                                      edgecolor='k', linewidths=1.,
                                                      zorder=2))
    
    #
    # Plot the floats verified to be in the north Atlantic ocean
    #
    goodlons = [longitudes[i] for i in in_natlantic]
    goodlats = [latitudes[i] for i in in_natlantic]
    x,y = map(goodlons, goodlats)
    map.scatter(x,y, s=5, zorder=5)
    
    #
    # Show
    #
    plt.show()

if __name__ == "__main__":
    main()