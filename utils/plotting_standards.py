#!/usr/bin/env python
import numpy as np

# bin edges and centers for the log10dt / charge plots

edges_dt, edges_Q = np.arange(-8.0,-1.0,0.1), np.arange(-0,6.,0.2)
X_dt, Y_Q = np.meshgrid(edges_dt,edges_Q)
x_center = edges_dt[0:-1]+(edges_dt[1]-edges_dt[0])/2.0
y_center = edges_Q[0:-1]+(edges_Q[1]-edges_Q[0])/2.0



# bin edges for the burst numbers and durations
edges_ppb, edges_bl = np.linspace(1,10,11), np.linspace(0.,3.,21)
X_ppb, Y_bl = np.meshgrid(edges_ppb,edges_bl)

# binning of the charge distribution
binning_charge  = np.linspace(0.0,3.,71)
binning_log10dt = np.arange(-8.0,-1.0,0.09)


# lineat plots of the delta-t distribution
raw_dt_bins = np.linspace(0.,0.01,201)
