#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 11:37:50 2018

@author: cassandra
"""
import glob
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

#
# Read out some aviso map of sla
#
directory = '/data/aviso_data/monthly_mean/'
files = glob.glob(directory+'*.hdf')
file = files[0]

df_sla = pd.read_hdf(file)

#
# Get out the time indexed map of sea surface height
#
#ssh_file = '/home/cassandra/Data/aviso_data/dataset-duacs-rep-global-merged-allsat-phy-l4_1532022114265.nc.hdf'
#df_ssh = pd.read_hdf(ssh_file)


#
# Get the sea level anomaly (in meters)
#

# What is this nv number? its either 0 or 1...
sla = df_sla.query('nv == 0')['sla'].unstack(1).values

#
# Retrieve lons and lats from the index
#
lons_sla = np.unique(df_sla.index.get_level_values('lon'))
lats_sla = np.unique(df_sla.index.get_level_values('lat'))



#
# Get the sea surface height (in meters)
#
#ssh = df_ssh.unstack(1).values


#
# Adjust the longitudes from [0, 360] to [-180, 180]
#
#west_lons = np.where(lons > 180.0)[0]
#for i in west_lons:
#    lons[i] = lons[i] - 360.0

#
# Generate a basemap
#
f = plt.figure(1)
map = Basemap()
#map.fillcontinents(color='#ddaa66',lake_color='aqua')
map.drawcoastlines()

#
# Adopt coordinates to map coordinates after forming a grid
#
#x, y = np.meshgrid(lons, lats)
#x, y = map(x,y)

#
# Plot the data!
#
map.pcolor(lons_sla, lats_sla, sla*100, cmap='coolwarm', latlon=True)

cbar = plt.colorbar(orientation='horizontal')
cbar.set_label('Sea Level Anomaly (cm)')

plt.title(file.split('/')[-1])
plt.tight_layout

plt.show()