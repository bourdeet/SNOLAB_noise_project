#!/usr/bin/env python

#######################################################
# pulse coincidence
# last update: April 4th 2018
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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import glob
import struct
import os
import subprocess
import readTrc_master.readTrc as trc


        
def mainprogram():

        
        parser = argparse.ArgumentParser(description="Analyze some coincidences",
                                         formatter_class=RawTextHelpFormatter)
        
        parser.add_argument('--inputdir',dest='INPUTDIR',
                            help="input directory containing the files",
                            required=True)

        parser.add_argument('--trigger',dest='TRIGGER',
                            type=int,
                            help="Channel number of the triggering DOM",
                            default=3
        )

        parser.add_argument('--readout',dest='READOUT',
                            type=int,
                            help="Channel number of the receiving DOM",
                            default=2
        )
        
        parser.add_argument('-o', '--output',dest='OUTFILE',
                            help="Generic name for an output file where the data is saved  (.p file)",
                            default="coincidence_results.p")
                
    
        parser.add_argument('--debug',dest='debug',
                            help='Enter debug mode: plots subsets of traces',
                            action='store_true')
 

        args = parser.parse_args()

        # Stuff happens now...
        #---------------------------------------------------------------------------------

        triggerformat = "C%i_*.trc"%(args.TRIGGER)
        readoutformat = "C%i_*.trc"%(args.READOUT)

        inputdir = args.INPUTDIR
        nprocessed = 0


        ncoincidence=0
        charge_trigger=[]
        charge_readout=[]
        for trace in sorted(glob.glob(inputdir+triggerformat)):
                
                triggerfile = trace.rstrip()
                
                if args.debug:
                        print "triggered file: ", triggerfile.split('/')[-1]
                        
                number = int((trace.split('_')[-1]).split('.')[0])
                proc = subprocess.Popen(['ls %s/C%i*%05i.trc'%(inputdir,args.READOUT,number)],
                                        stdout=subprocess.PIPE,shell=True)
                
                readoutfile = proc.stdout.read().rstrip()
                if args.debug:
                        print "readout   file: ",readoutfile.split('/')[-1] 

                # Readout the trace
                #-----------------------------------------------------------------
                data_trigger = trc.readTrc(triggerfile)
                X = data_trigger[0]
                trigger_times = data_trigger[2]
                trigger_meta= data_trigger[3]

                # Adjust the time of the vector to  the individual triggered segments
                # The timing is the exact same for the readout wavelength
                
                trace_length = trigger_meta['WAVE_ARRAY_COUNT']/trigger_meta['SUBARRAY_COUNT']
                adjusted_time = (np.arange(0,len(X))%trace_length)*trigger_meta['HORIZ_INTERVAL']
                trigtime_mapping = np.repeat(trigger_times['trigtime'],trace_length)  
                offset_mapping = np.repeat(trigger_times['offset'],trace_length)
                adjusted_time = adjusted_time+trigtime_mapping+offset_mapping


                # Record the file trigger time to compute the approximate livetime of the run
                #------------------------------------------------------------------------------
                if nprocessed==0:
                        start_time = trigger_meta['TRIGGER_TIME']
                else:
                        end_time = trigger_meta['TRIGGER_TIME']

                nprocessed+=1

                
                # Proceed to readout the receiving DOM
                #--------------------------------------------------------------------
                
                data_readout = trc.readTrc(readoutfile)
                
                if args.debug:
                        print "DEBUG MODE"
                        for i in range(20,150):
                                a = trace_length*i
                                b = a+trace_length
                                midpoint = a+trace_length/2
                                plt.plot(adjusted_time[a:b],data_trigger[1][a:b],'o',label='trigger')
                                plt.plot(adjusted_time[a:b],data_readout[1][a:b],label='readout')
                                plt.plot([adjusted_time[midpoint],adjusted_time[midpoint]],[-0.01,0.005],'r')
                                plt.show()
                                
                readout_background = np.median(data_readout[1])
                
                for i in range(0,trigger_meta['SUBARRAY_COUNT']):
                        a = trace_length*i
                        b = a+trace_length
                        start_integration = a+trace_length/2-3
                        stop_integration = start_integration+15  # we restrict pulse search in the first 40ns after trigger
                        integration_length = stop_integration-start_integration

                        Q_trigger = -sum(data_trigger[1][start_integration:stop_integration])
                        charge_trigger.append(Q_trigger)
                        Q_readout = -sum(data_readout[1][start_integration:stop_integration])

                        # Record a coincidence only if 

                        if Q_readout>(-3.5*readout_background*integration_length): # has been optimized to eliminate bad counts

                                if args.debug:
                                        plt.plot(adjusted_time[a:b],data_trigger[1][a:b],label='trigger')
                                        plt.plot(adjusted_time[a:b],data_readout[1][a:b],label='readout')
                                        plt.show()
                                        
                                ncoincidence+=1
                                charge_readout.append(Q_readout)
                        
                print ncoincidence
                        
                if args.debug:
                        plt.hist(charge_trigger,bins=30)
                        plt.show()
                        plt.hist(charge_readout,bins=30)
                        plt.show()

        print "The beginning of the run: ",start_time
        print "The end of the run: ",end_time
        print "The coincidence rate: ",float(ncoincidence)/((end_time-start_time).total_seconds())
                        
if __name__=='__main__':
        mainprogram()
