#!/usr/bin/env python

#######################################################
# pulse analyzer
# last update: May 1st 2018
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
        self.mode = 'unspecified'
        self.nacq=0
        self.data = {'start':1,'stop':1}
        self.time ={'scale':200e-6,'duration':10e-6,'vec':[]}
        self.triglvl=0
        self.impedance=50.0

    def __repr__(self):
        return "Window Scale: {}\n# of acquisitions: {}\nData Start: {}\nData Stop: {}\nTime parameters: {}\nTrigger level (V): {}".format(self.windowscale, self.nacq,self.data['start'],self.data['stop'],self.time,self.triglvl)
        
        
class PMT_DAQ_sequence:
    def __init__(self):
        self={'charge':[],'time':[],'livetime':0.0,'npulses':0,'pedestal':0.0,'mode':'unspecified'}

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
            threshold=-threshold
        
    N=len(data)
    lowbound=min(data)
    bound=max(data)
    VETO=5
    
    Q=[]
    t=[]
    q=0
    qmax=0
    tmax=0

    impedance=header.impedance
    
    ispulse=False
    
    integrate=False
    
    veto=False

    
    integ=[]
    vetolength=[]
    ispulsevec=[]
    timegate=np.zeros(len(data))
    timeint=header.windowscale['xincr']/1e-9
    
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
                    Q.append(q*timeint/impedance*1000)
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
    
    return -np.array(Q),np.array(t)*timeint


def find_pulses_array(X,Y,D,sequence_time=None,threshold=-0.1,Nsample=5,debug=False):

        print Nsample

        #***************  This code assumes a negative pulse convention  ***************

        if sequence_time is None:
                time = X

        else:
                # if the data comes from a sequence acquisition,
                # we expect that the user supplies an adjusted time
                # array based on the triggering offsets
                time = sequence_time

        # Acquire information from the header:
        sample_res_ns = D['HORIZ_INTERVAL']/1e-9
        impedance = float(D['VERT_COUPLING'].split('_')[1])

        if debug:
                plt.plot(time,Y,'r')#[0:10000],Y[0:10000],'r')
                plt.title('data, adjusted timing')
                plt.show()

        # Select data below threshold (must be done BEFORE removing pedestal)
        #-------------------------------------------------------------------

        signal_mask = Y<=threshold 
        signal   = Y*(signal_mask)

        
        if debug:
                plt.plot(time,signal,'r')#[0:10000],signal[0:10000],'r')
                plt.title('data below threshold')
                plt.show()

                
        # Compute and subtract pedestal
        #--------------------------------------------------------------------
        
        pedestal = Y[~signal_mask]
        if debug==True:
                plt.hist(pedestal,bins=30)
                plt.yscale('log')
                plt.title("value of the non-signal")
                plt.show()
                
        pedestal = np.median(pedestal)
        signal = signal-pedestal*(signal_mask)
        
        if debug==True:
                plt.plot(time,signal,'orange')#[0:10000],signal[0:10000],'orange')
                plt.title('data below threshold, minus pedestal')
                plt.show()

        # Clean signal mask
        #--------------------------------------------------------------------

        
        # append a zero at the beginning and end of the signal vector
        # (to make sure we finish all pulses)
        signal = np.concatenate([[0],signal,[0]])
        signal_mask = np.concatenate([[False],signal_mask,[False]])

        # Find the location of the threshold crossings in the array
        diff = np.diff(signal_mask)

        index, = diff.nonzero()
        start = index[:-1]
        intervals = index[1:]-index[0:-1]
        true_intervals = intervals*(np.arange(0,len(intervals))%2==0)
        
        # Select signal regions larger than a certain number of samples
        true_intervals=true_intervals*(true_intervals>=Nsample)

        pt=[]

        for i in range(0,len(start)):

                if true_intervals[i]!=0:

                        pt.append(np.arange(start[i]+1,start[i]+true_intervals[i]+1))

        
        pt = np.concatenate(pt)

        selected_pulse=np.zeros(len(signal_mask))

        selected_pulse[pt]=True

        if debug==True:
                plt.plot(time,Y,'r')
                plt.plot(time,-0.005*selected_pulse[1:-1])
                plt.plot(time,Y*selected_pulse[1:-1],'go')
                plt.title("Pulses selected")
                plt.show()

        # Now, selected_pulse is the cleaned signal mask that only
        # keeps pulses that pass the threshold, and have a minimal width
        # of N samples


        pulses = np.split(signal*selected_pulse*sample_res_ns/impedance*1000,start+1)

        print start+1

        if debug==True:
                for element in pulses:
                        if sum(element)!=0:
                                print "charge of this pulse:",sum(element)
                                print "length of this pulse: ",len(element)
                                plt.plot(element)
                                plt.show()
                        
                        


        charge = np.array([sum(x) for x in pulses])
        charge = charge[charge!=0]

        # The pulse time tag is defined as the sample time of the first
        # data point crossing the selected threshold
        
        time_indices  = start[np.in1d(start+1,selected_pulse.nonzero())]+1
        times = time[time_indices]


        # Remove the last item which is probably bad
        charge = charge[:-1]
        times = times[:-1]
        
        return -charge,times


def find_pulses_flasherrun(X,Y,D,interval=[20,40],threshold=50.0,debug=False):

        #***************  This code assumes a negative pulse convention  ***************

        # In a flasher run, we don't care about the time, because the trigger is external

        # Acquire information from the header:
        sample_res_ns = D['HORIZ_INTERVAL']/1e-9
        impedance = float(D['VERT_COUPLING'].split('_')[1])

        
        trace_length = D['WAVE_ARRAY_COUNT']/D['SUBARRAY_COUNT']
        
        mask = np.arange(1,D['WAVE_ARRAY_COUNT']+1)

        mask = (mask%trace_length>=interval[0])&(mask%trace_length<=interval[1])


        signal = Y*mask
        background = Y[~mask]

        if debug:
                
                plt.hist(background,bins=20)
                plt.title('Pedestal distribution')
                ax = plt.gca()
                plt.text(0.3, 0.8,'median = %f'%np.median(background), horizontalalignment='center',
                         verticalalignment='center', transform=ax.transAxes, fontsize=12)
                plt.show()
                
                X2 = np.arange(0,len(X))%trace_length
                
                plt.plot(X2[0:5000],signal[0:5000],color='orange')
                plt.plot([0,trace_length],[0,0],'k',linewidth=2.0)
                plt.title('Raw signal (stacked)')
                ax = plt.gca()
                plt.text(0.7, 0.8,'median = %f'%np.median(signal), horizontalalignment='center',
                         verticalalignment='center', transform=ax.transAxes, fontsize=12)
                plt.show()
                
                
        pedestal = np.median(background)
        signal = (signal-pedestal)*mask

        if debug:
                
                plt.plot(X2[0:5000],signal[0:5000],color='red')
                plt.plot([0,trace_length],[0,0],'k',linewidth=2.0)
                plt.title('pedestal-subtracted signal (stacked)')
                ax = plt.gca()
                plt.text(0.7, 0.8,'median = %f'%np.median(signal), horizontalalignment='center',
                         verticalalignment='center', transform=ax.transAxes, fontsize=12)
                plt.show()
        
        charge = []

        pulse_change = np.diff(mask)
        start = np.where(pulse_change)[0]

        # pick only the even elements of the array
        start = start[(np.arange(0,len(start))%2==0)]

        charge = []
        dq = interval[1]-interval[0]

        for s in start:
                charge.append(-sum(signal[s:s+dq]*sample_res_ns/impedance)*1000)


        #plt.hist(charge,bins=100)
        #plt.yscale('log')
        #plt.xlabel('charge (pC)')
        #plt.show()

        return charge
