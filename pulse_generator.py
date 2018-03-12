#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

def add_noise(Y,poi):
    y2=np.zeros(len(Y))

    for i in range(len(Y)):
        noise=np.random.poisson(poi)
        y2[i]=Y[i]+noise

    return y2
    
def insert_random_pedestal(poi):

    N=np.random.randint(0,100)
    B= np.random.poisson(poi,N)

    return B.astype(float)

def generate_pulse(x,A,sigma,alpha,to=40):
    y=np.zeros(len(x))
    for i in range(0,len(x)):

        if x[i]<=to:
    
            y[i]=A*np.exp(-(x[i]-to)**2/sigma**2)

        else:

            y[i]=A*np.exp(-(x[i]-to)**2/(sigma**2+alpha*(x[i]-to)))


    return y

def add_EM_interference(data,freq=60.0,resolution=2e-9):
    newdata=[]
    amplitude=3.
    
    for i in range(len(data)):
        newdata.append(data[i]+np.sin(2.0*np.pi*freq*float(i)*resolution)*amplitude)

    return newdata


import argparse

parser = argparse.ArgumentParser(description="Generates fake PMT pulse data.")

parser.add_argument('-N',type=int,default=1000, dest='N',help="Number of pulses to generate.")

parser.add_argument('-P', '--poi',default=5, dest='P',help="Poisson noise rate (in arbitrary count)")

parser.add_argument('-A', '--amplitude',type=float,default=10,dest='A',help="single pe amplitude")

parser.add_argument('-S', '--size',default=200,dest='S',help="Size of the acquisition window, in sample number")

parser.add_argument('-r', '--resolution',type=float,default=2.0*1e-9,dest='RES',help="Resolution of one sample in seconds.")

parser.add_argument('-f', '--file',default=None,dest='FILE',help="If selected, saves to trace to a pickle file.")

args = parser.parse_args()


# ---------------------------------------------------------------
# Demonstration of the pulse making process

x=np.linspace(0,100,200)
y=generate_pulse(x,50.0,5,15)
y2=add_noise(y,5)
plt.plot(y)
plt.show()
plt.plot(y2)
plt.show()



# number of pulses to generate
N=args.N

# Poisson noise rate
poi=args.P

# Single pe amplitude
pe=args.A

# amplitude distribution of the pulses
As=np.random.uniform(0.5*pe,10*pe,N)

# sigma distributions of the 
sigmas=0.3*pe#np.random.uniform(0.3*pe,pe,N)

# alpha distributions
alphas=0.7*pe#np.random.uniform(0.5*pe,3*pe,N)

datastream=[]
timestamp=[]

endpoint=args.S/args.RES #the endpoint in nanoseconds

x2=np.linspace(0,args.S,args.S)
m=0
for i in range(N):


    noisy=[]#insert_random_pedestal(poi)

    signal=generate_pulse(x2,As[i],sigmas,alphas)#sigmas[i],alphas[i])
    signal_plus_background=add_noise(signal,poi)

    for j in range(len(noisy)):
        m+=1
        datastream.append(noisy[j])
        timestamp.append(float(m*args.RES))
        
    for j in range(len(signal_plus_background)):
        m+=1
        datastream.append(signal_plus_background[j])
        timestamp.append(float(m*args.RES))


total=np.asarray(datastream)

#add an EM interference signal of 60Hz to the entire data set
total_EM=add_EM_interference(total,60.0,args.RES)

print m
print i

if args.FILE==None:
    print "User doesn't want to save this data stream."
    plt.plot(timestamp,total_EM)
    plt.xlabel("Time (s)")
    plt.show()
    sys.exit()

elif isinstance(args.FILE,basestring):
    print "Saving trace as pickle file: ",args.FILE
    header={}
    header['N']=args.N
    header['res']=args.RES
    header['poisson']=args.P
    header['size']=args.S
    header['spe']=args.A
    pickle.dump([header,timestamp,total_EM],open(args.FILE,"wb"))

else:
    print " I don't understand the voodoo magic you wrote. hej Hej!"
    sys.exit()
