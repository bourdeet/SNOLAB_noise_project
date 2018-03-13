#!/usr/bin/env python

#######################################################
# pulse analyzer
# last update: September 6th 2017
#
# Author: Etienne Bourbeau
#         (etienne.bourbeau@icecube.wisc.edu)
#
#
#######################################################
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import glob
import struct
import os

class header_data:
    def __init__(self):

        self.windowscale = {'ymult':0.0,'yzero':0.0,'yoffs':0.0,'xincr':0.0}
        self.nacq=0
        self.data = {'start':1,'stop':1}
        self.time ={'scale':200e-6,'duration':10e-6,'vec':[]}
        self.triglvl=0

    def __repr__(self):
        return "Window Scale: {}\n# of acquisitions: {}\nData Start: {}\nData Stop: {}\nTime parameters: {}\nTrigger level (V): {}".format(self.windowscale, self.nacq,self.data['start'],self.data['stop'],self.time,self.triglvl)
        
        
class PMT_DAQ_sequence:
    def __init__(self):
        self={'charge':[],'time':[],'livetime':0.0,'npulses':0,'pedestal':0.0}

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.keys())

    def __unicode__(self):
        return unicode(repr(self.__dict__))

def compute_pedestal(trace, n=1000):
    
    n_per_slice=len(trace)/n
    pedestal=[]
    for i in range(0,n):
        pedestal.append(np.median(trace[i*n_per_slice:(i+1)*n_per_slice]))

    up = np.mean(pedestal)+5*np.std(trace)
    down= np.mean(pedestal)-5*np.std(trace)
    
    A=trace>down
    B=trace<up
    C = [all(f) for f in zip(A,B)]
    fives_threshold=(1-(len(C)-sum(C))/float(len(C)))*100
    #print "Pedestal 5-sigma-ish containement: ",fives_threshold," \%"

    return np.mean(pedestal),np.std(trace),fives_threshold


def find_pulses_in_that_shit(header,data,threshold=0.1,Inverted=False,debug=False):

    # The code expects positive, pedestal-subtracted pulses.
    # To feed in negative value simply switch the inverted boolean

    if Inverted==True:
        data=-data
        
    N=len(data)
    lowbound=min(data)
    bound=max(data)
    VETO=5
    
    Q=[]
    t=[]
    q=0
    qmax=0
    tmax=0
    
    ispulse=False
    
    integrate=False
    
    veto=False

    
    integ=[]
    vetolength=[]
    ispulsevec=[]
    timegate=np.zeros(len(data))
    
    pulsesize=0
    
    for i in range(len(data)):
        
                
        if(ispulse==False): #if previous data point was below threshold

            if data[i]>threshold: # and new data point was above
                
                if not veto:   # first crossing of threshold outside veto
                    integrate=True
                    ispulse=True
                    veto=True

                else: # data crossing occurred during a veto. We don't care.
                    pulsesize+=1


            else: # previous data was below and this data is below. do nothing
                
                if veto:
                    pulsesize+=1

        else: # the previous data point was abov threshold
            
            if data[i]<threshold: #The next data point is below

                if not veto: # first downward crossing. stop integrating
                    
                    integrate=False
                    ispulse=False
                    pulsesize=0
                    veto=True
                    Q.append(q)
                    t.append(tmax)
                    
                    q=0
                    tmax=0
                    qmax=0

                else: # Simply increment veto
                    pulsesize+=1

            

        # check veto status: reset if need be
        if veto and pulsesize>=VETO:
            pulsesize=0
            veto=False

        # Check integration status:
        if integrate:
            integ.append(bound)
            q+=data[i]
            if data[i]>qmax:
                qmax=data[i]
                tmax=i
        else:
            integ.append(0)


        if veto:
            vetolength.append(bound*0.5)
        else:
            vetolength.append(0)
            
        if ispulse:
            ispulsevec.append(bound*0.75)
        else:
            ispulsevec.append(0)
    
    if debug:

        time_location=np.zeros(len(header.time['vec']))
        time_location[t]=bound

        if len(data)>200000:
            pt = 200000
        else:
            pt = len(data)
        beg = np.where(data==max(data))[0][0]
        print beg
        plt.plot(header.time['vec'][(beg-100000):(beg+100000)],data[(beg-100000):(beg+100000)])
        plt.plot(header.time['vec'][(beg-100000):(beg+100000)],time_location[(beg-100000):(beg+100000)],'r')
        plt.plot(header.time['vec'][(beg-100000):(beg+100000)],vetolength[(beg-100000):(beg+100000)],'g')
        plt.plot(header.time['vec'][(beg-100000):(beg+100000)],ispulsevec[(beg-100000):(beg+100000)],'k')
    
        plt.xlabel("time(s)")
        plt.ylabel("signal(V)")
        plt.title("pulse_location")
        plt.show()
    
    return Q,t
