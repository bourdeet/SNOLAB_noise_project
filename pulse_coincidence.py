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



        
def mainprogram():

        
        parser = argparse.ArgumentParser(description="Analyze some coincidences",
                                         formatter_class=RawTextHelpFormatter)
        
        parser.add_argument('--inputdir',dest='INPUTDIR',
                            help="input directory containing the files",
                            required=True)

        parser.add_argument('--trigger',dest='TRIGGER',
                            type=int,
                            help="Channel number of the triggering DOM",
                            default=2
        )

        parser.add_argument('--readout',dest='READOUT',
                            type=int,
                            help="Channel number of the receiving DOM",
                            default=3
        )
        
        parser.add_argument('-o', '--output',dest='OUTFILE',
                            help="Generic name for an output file where the data is saved  (.p file)",
                            default="coincidence_results.p")
                
    
        parser.add_argument('--debug',dest='DEBUG',
                            help='Enter debug mode: plots subsets of traces',
                            action='store_true')


        args = parser.parse_args()

        # Stuff happens now...
        #---------------------------------------------------------------------------------

        triggerformat = "C%i_*.trc"%(args.TRIGGER)
        readoutformat = "C%i_*.trc"%(args.READOUT)

        inputdir = args.INPUTDIR
        
        for trace in sorted(glob.glob(inputdir+triggerformat)):

                print trace
                number = int((trace.split('_')[-1]).split('.')[0])

                #os.system("ls %s/C%i*%i.trc"%(inputdir,args.READOUT,number))
                continue
                #sys.exit()
                
                # Readout the trace

                # Locate the pulses in the trace

                # save the time of these pulses

                #for readout in 

                #check if there's something in the readout

                if something:
                        coincidence+=1
                else:
                        coincidence=0
                        #nothing
        
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
