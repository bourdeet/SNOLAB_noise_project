#!/usr/bin/env python


from pulsetools import *
import pickle
import numpy as np


n_series = 1000
livetime = 50.e-3

Poisson_rate = 1700.


seq_info=PMT_DAQ_sequence()
list_of_seq = []
#
#Magic here
for i in range(0,n_series):
    
    seq_info=PMT_DAQ_sequence()

    N_pulses_in_series = np.random.poisson(Poisson_rate*livetime)

    times = np.random.uniform(0,50e-3,N_pulses_in_series)
    charge = np.random.normal(1.0,0.8,N_pulses_in_series)

    times = np.sort(times)


    seq_info['charge']=charge
    seq_info['time']=times
    seq_info['livetime']=livetime
    seq_info['npulses']=N_pulses_in_series
    seq_info['mode']='normal'

    list_of_seq.append(seq_info)

    del seq_info

pickle.dump(list_of_seq,open("poisson_data.p","wb"))

    
