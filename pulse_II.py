#!/usr/bin/env python

#######################################################
# pulse II
# last update: September 27th 2017
#
# Author: Etienne Bourbeau
#         (etienne.bourbeau@icecube.wisc.edu)
#
# scripts that work on previously saved pickle file of
# an oscilloscope run
#
#######################################################


from pulsetools import *
import pickle
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="pulse II",formatter_class=RawTextHelpFormatter)
parser.add_argument('-i', '--input',dest='INFILE', help="Input Data (pickle file)",required=True)

args = parser.parse_args()

data = pickle.load(open(args.INFILE,"rb"))

charge=[]
deltatees=[]
i=0

npulses=[]
livetime=[]

for sequence in data:

    #if isintance(sequence,header_data):
        #timescale=sequence.

    # Get rid of a nested list problem
    if isinstance(sequence,list):
        sequence=sequence[0]

    if isinstance(sequence,PMT_DAQ_sequence):
        npulses.append(float(sequence['npulses'])-1)
        livetime.append(sequence['livetime'])
        for q in sequence['charge']:
            if q>2000:
                print sequence
            else:
                charge.append(q)

        if sequence['npulses']>1:
            t = np.asarray(sequence['time'])*2.0e-9
            dT = np.log10(t[1:]-t[:-1])
            for element in dT:
                deltatees.append(element)

#plt.yscale('log', nonposy='clip')
plt.ylabel("count")
plt.xlabel("charge (a.u.)")
plt.hist(charge,bins=100,range=[0,0.1])
plt.show()

plt.figure(2)
#print np.log10(6.0e-6) #imitate the artificial deadtime circuit used in PMT calibration paper
plt.ylabel("count")
plt.xlabel("log10(dT) (s)")

with open("../analysis_data/Hitspool_2014_2017_dom05-05_example.p","rb") as hitspool:

    HS14,HS17=pickle.load(hitspool)

    HS14=np.asarray(HS14)
    #condition=(HS14>=-5.6)&(HS14<=-3)
    #HS14=HS14[condition]
    
    #W=HS14/np.log10(1.0e-3)/float(len(HS14))*(np.log10(1e-3)-HS14)
    W=np.array([1/float(len(HS14))]*len(HS14))

    plt.hist(HS14,bins=100,range=[-8,-1],alpha=0.5,label="Hitspool 2014",weights=W)

deltatees=np.asarray(deltatees)
#condition=(deltatees>=-5.6)&(deltatees<=1.e-3)
#deltatees=deltatees[condition]

V=np.array([1/float(len(deltatees))]*len(deltatees))

plt.hist(deltatees,bins=100,range=[-8,-1],alpha=0.5,label="SNOLab test run",weights=V)#,weights=np.array([1.0/float(len(deltatees))]*len(deltatees)))

plt.legend()

plt.show()

N=sum(np.asarray(deltatees)>=np.log10(6e-6))

print "pulse Rate for this file: ",N/sum(livetime)," Hz"
