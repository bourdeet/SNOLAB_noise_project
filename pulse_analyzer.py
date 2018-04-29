#!/usr/bin/env python

#######################################################
# pulse analyzer
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
import matplotlib.pyplot as plt
import glob
import struct
import os

#from pulsetools import 


def FFT_that_shit(header,data):

    F=np.fft.fft(data,n=len(data))
    x=np.fft.fftfreq(n=len(data),d=header['res'])
    plt.plot(x,F)
    plt.xlim([0,200])
    plt.show()
    return x,F

def plot_that_shit(header,data,timestamp,option=0,n=100):
    
    resolution=header['res']
    nsample=header['size']
    plt.figure()
    plt.xlabel("Time (s)")
    plt.ylabel("Signal (mV)")

    if option==0: # plot all
        plt.plot(timestamp,data)
        plt.show()
        
    elif option==1: # plot only subset
        plt.plot(timestamp[:n],data[:n])
        plt.show()
        
    elif option==2:
        x=[i%(n*resolution) for i in timestamp]
        plt.plot(x,data)
        plt.show()
    else:
        print "ERROR: unrecognized option"
        


        
def mainprogram():

        if True:
                parser = argparse.ArgumentParser(description="Analyze some pulses",formatter_class=RawTextHelpFormatter)
                parser.add_argument('-i', '--input',dest='INFILE',nargs='*',\
                                    help="Input Data - can be any of the following:\n\t-Single string\n\t-list of strings\n\t-String pattern to search for\n",\
                                    required=True)

                parser.add_argument('-o', '--output',dest='OUTFILE',\
                                    help="Generic name for an output file\n where the data is saved  (.p file)",\
                                    default="summary.p")
                
                parser.add_argument('-t','--threshold',dest='THRESH',
                                    help="Exclude pulses below a user-defined threshold (in V)",
                                    type=float,
                                    default=-0.0025)
    
                parser.add_argument('--debug',dest='DEBUG',
                                    help='Enter debug mode: plots subsets of traces',
                                    action='store_true')


                parser.add_argument('--flash-as-seq',dest='FasS',
                                    help='Process flasher runs as sequence runs',
                                    action='store_true')
                
                parser.add_argument('--asFlash',dest='SasF',
                                    help='Process sequence runs as flasher runs',
                                    action='store_true')
                
                parser.add_argument('--interval',dest='INTERVAL',
                                    help='time window interval for flasher run',
                                    default ="20,40"
                )

                parser.add_argument('--pulse_width',dest='WIDTH',
                                    help='number of samples to integrate charge over',
                                    type=int,
                                    default =3
                )
                
                args = parser.parse_args()

                interval = [int(args.INTERVAL.split(',')[0]),int(args.INTERVAL.split(',')[1])]
                
                if args.DEBUG:
                        print 'there\'s gonna be some debugging happenin...'
            
                #Welcome message
                print "**************************************\n\tWelcome to waveform analyzer\n"
                print "**************************************"

                print "checking your input arguments...\n"
                p=0
                nsequences=0
                if args.INFILE!=None and args.INFILE !='':
                        filelist=args.INFILE
                        print "the filelist: ",filelist
        
                        if len(args.INFILE)==1:
                                print "analyzing a single file: ",args.INFILE,"...\n"
            
                        elif len(args.INFILE)>1:
                                print "analyzing multiple files...\n "
            

                        filetype=filelist[0]


                        if filetype.endswith(".bin"):
                                print "\nDetected a binary file."
                                print "Importing relevant libraries..."

                                from DPO3200bin_tools import parse_header,load_data_bin

                                print "Searching corresponding header file..."
                                genericname=filetype[:-8]
                                headerfile=glob.glob(genericname+'header.txt')
        
                                if len(headerfile)<1:
                                        sys.exit("ERROR: could not find a matching header file.")

                                else:
                                        print "Found! Parsing the header.."
                                        header=parse_header(headerfile[0],len(filelist))

                                if args.DEBUG:
                                        header.nacq=10
                                        sequence_length=header.data['stop']-header.data['start']+1

                
                                for element in filelist:
                                        if element.endswith(".bin"):
                    
                                                if os.path.exists(args.OUTFILE):
                                                        pulses_info=pickle.load(open(args.OUTFILE,"rb"))
                                                else:
                                                        pulses_info=[]
                        
                                                newinfo=load_data_bin(element,header)
                                                pulses_info=pulses_info+newinfo
                                                pickle.dump(pulses_info,open(args.OUTFILE,"wb"))
                                        
                                                nsequences+=len(newinfo)
                    
                                print "Saved a total of ",nsequences," sequences to pickle files"
            

                        elif filetype.endswith(".trc"):
                                print "\nDetected a LeCroy-formatted binary (.trc)"
                                print "Importing relevant libraries..."

                        
            
                                from trc_tools import load_data_trc
                                print "...done."
                                nfiles = 0
                               
                                
                                for element in filelist:
                                        if element.endswith(".trc"):
                                                nfiles+=1
                                                if nfiles%10==0:
                                                        print "processed %i files..."%(nfiles)

                                                X = element.split('_')[-1][0:5]#[3:8]
                                                filennum = int(X)

                                                # Save one pickle file per input trc file
                                                newinfo,header=load_data_trc(element,threshold=args.THRESH,
                                                                             interval=interval,
                                                                             asSeq = args.FasS, #treat flasher runs as sequence runs
                                                                             asFlash=args.SasF,
                                                                             debug=args.DEBUG,
                                                                             Nsample = args.WIDTH
                                                )
                                                # pickle.dump(header,open(args.OUTFILE[:-2]+"_header.p","wb"))
                                                # Dump data
                                                pickle.dump(newinfo,open(args.OUTFILE[:-2]+"_%05i.p"%filennum,"wb"))

                                        del header,newinfo
                    
                                """
                                elif filetype.endswith(".csv"):
                                from otherfiles_tools import *
            
                                for element in filelist:
                                        H,T,Y=load_data_csv(element)
                
                                elif filetype.endswith(".p"):
                                from otherfiles_tools import *
                                for element in filelist:
                                        H,T,Y=load_data_pickle(element)
                                """
                        else:
                                print "******ERROR******\nI don't recognize your voodoo magic data type."
                                sys.exit(-1)

                else:
                        print "ERROR. no file entered."

                        
if __name__=='__main__':
        mainprogram()
