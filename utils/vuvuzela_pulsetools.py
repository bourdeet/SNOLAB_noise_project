#!/usr/bin/env python

#######################################################
# pulsetools for vuvuzela data
#
# Simplified library from the SNOLab project
# Author: Etienne Bourbeau
#         (etienne.bourbeau@icecube.wisc.edu)
#
#
#######################################################

import numpy as np

from pulsetools import PMT_DAQ_sequence
from pulsetools import compute_pedestal
from pulsetools import gauss


def find_pulses_array(X,Y,D,sequence_time=None,threshold=-0.1,Nsample=10,debug=False,n=0):
    '''
    slightly modified version of the pulse finder that does not use impedance info

    '''

    
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
    #print "vertical: ",D['VERTUNIT']
    
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
        plt.plot(time,signal,'bo')
        plt.title('data below threshold')
        plt.show()

                
    # Compute and subtract pedestal
    #--------------------------------------------------------------------
        
    pedestal = Y[~signal_mask]
    median = np.median(pedestal)
    ped_sigma = np.std(pedestal)

    
    if debug:

        from scipy.optimize import curve_fit

        hist, bin_edges = np.histogram(pedestal,bins=25)
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
        
        p0 = [len(pedestal), np.mean(pedestal), np.std(pedestal)]
        coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
        hist_fit = gauss(bin_centres, *coeff)
        
        plt.hist(pedestal,bins=25)
        plt.plot(bin_centres,hist_fit,'r')
        plt.plot([coeff[1]-4*coeff[2],coeff[1]-4*coeff[2]],[0.,2000000],'g',linewidth=2.)
        plt.plot([coeff[1]+4*coeff[2],coeff[1]+4*coeff[2]],[0.,2000000],'g',linewidth=2.)
        #plt.yscale('log')
        plt.title("value of the non-signal")

        print "\n\n\n***********************\n\n"
        print "Pedestal cut threshold: ",coeff[1]-4*coeff[2]
        print "\n\n***********************\n\n\n"
        plt.show()


    pedestal = np.median(pedestal)
    signal = signal-pedestal*(signal_mask)
    
    if debug:
        plt.plot(time,signal,'orange')#[0:10000],signal[0:10000],'orange')
        plt.plot(time,signal,'bo')
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

    if debug:
        plt.plot(time,Y,'r')
        plt.plot(time,-0.005*selected_pulse[1:-1])
        plt.plot(time,Y*selected_pulse[1:-1],'go')
        plt.title("Pulses selected")
        plt.show()

            
    # Now, selected_pulse is the cleaned signal mask that only
    # keeps pulses that pass the threshold, and have a minimal width
    # of N samples


    pulses = np.split(signal*selected_pulse,start+1)

    #print sample_res_ns,impedance
    #print start+1

    # print all pulses infividually
    # Save them in a file if need be (for pulse shape averaging)
    if debug:
        max_samp = 20
        traces = []
        for element in pulses:
            element = np.concatenate([[0.0],element])
            
            if element.size<max_samp:
                element = np.concatenate([element,np.zeros(max_samp-element.size)])
            elif element.size>max_samp:
                element = element[:max_samp]
                
                
            if sum(element)!=0:
                print "charge of this pulse:",sum(element)
                print "length of this pulse: ",len(element)
                plt.plot(element,'bo')
                plt.plot(element,'g')
                #traces.append(element)
        #import pickle
        #pickle.dump(traces,open("list_of_traces_%i.p"%n,"wb"))
        #print "saved traces in list_of_traces.p"
                        
                        


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
