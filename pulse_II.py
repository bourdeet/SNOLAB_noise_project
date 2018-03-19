#!/usr/bin/env python

#######################################################
# pulse II
# last update: March 13th 2018
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

parser = argparse.ArgumentParser(description="pulse II",
                                 formatter_class=RawTextHelpFormatter)

parser.add_argument('-i', '--input',
                    dest='INFILE',
                    help="Input Data (pickle file)",
                    required=True)

parser.add_argument('--input2',
                    dest='INFILE2',
                    help="second set of data")


args = parser.parse_args()


# Load data containers
#------------------------------------------------------------------

charge=[]
deltatees=[]
npulses=[]
livetime=[]

# Load header information
#------------------------------------------------------------------

for pickled_file in glob.glob(args.INFILE):
    
    if 'header' not in pickled_file:
        

        data = pickle.load(open(pickled_file,"rb"))


        for sequence in data:

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
                                        charge.append(-q)

                        if sequence['npulses']>1:
                                t = np.asarray(sequence['time'])
                                dT = np.log10(t[1:]-t[:-1])
                                for element in dT:

                                       deltatees.append(element)


# Load second dataset if it exists
#-----------------------------------------------------------------------
charge_II=[]
deltatees_II=[]
npulses_II=[]
livetime_II=[]

if args.INFILE2 is not None:



        for pickled_file in glob.glob(args.INFILE2):
    
                if 'header' not in pickled_file:
        

                        data = pickle.load(open(pickled_file,"rb"))


                for sequence in data:


                        # Get rid of a nested list problem
                        if isinstance(sequence,list):
                    
                                sequence=sequence[0]

                        if isinstance(sequence,PMT_DAQ_sequence):
                
                                npulses_II.append(float(sequence['npulses'])-1)
                                livetime_II.append(sequence['livetime'])
                
                                for q in sequence['charge']:
                                        if q>2000:
                                                print sequence
                                        else:
                                                charge_II.append(q)

                                if sequence['npulses']>1:
                                        t = np.asarray(sequence['time'])
                                        dT = np.log10(t[1:]-t[:-1])
                                        for element in dT:
                                                deltatees_II.append(element)




                        
# Plotting
#-----------------------------------------------------------------------


# Define the binning for delta-t histogram comparison
binning = np.arange(-8.0,-1.0,0.1)

binning_charge = np.arange(0,0.1,0.001)


# Charge distribution
plt.ylabel("count")
plt.xlabel("charge (pC)")
plt.yscale('log')
y,x,_=plt.hist(charge,bins=100,color='g',label='charge method 1',alpha=0.5)

dx = (x[1:]-x[0:-1])[0]
x = x+dx/2



# Fitting spe peaks
#-------------------------------------------------------------------------

def gaussian(x,mu,sigma):
        return 1./(sigma*np.sqrt(np.pi))*exp(-((x-mu)**2.)/(2.*sigma**2.0))

def SPE(x,mu_ped,s_ped,mu_exp,mu_1pe,s_1pe,n_pe_max=8):

        bkgd = gaussian(x,mu_ped,s_ped)*exp(-mu_exp)

        signal = 0.0
        for i in range(0,n_pe_max):
                signal+=gaussian(x,i*mu_1pe,np.sqrt(s_1pe))*poi(i,mu_exp)
        
        return signal+bkgd


#plt.hist(charge_II,bins=100,color='r',label='charge method 2',alpha=0.5)
plt.legend()
plt.show()

plt.figure(2)
plt.ylabel("count")
plt.xlabel("log10(dT) (s)")


with open("../analysis_data/Hitspool_2014_2017_dom05-05_example.p","rb") as hitspool:

    HS14,_=pickle.load(hitspool)

    HS14=np.asarray(HS14)

    W=np.array([1/float(len(HS14))]*len(HS14))

    plt.hist(HS14,bins=binning,range=[-8,-1],alpha=0.5,label="Hitspool 2014",weights=W)

    
deltatees=np.array(deltatees)
#deltatees_II=np.array(deltatees_II)

V=np.array([1/float(len(deltatees))]*len(deltatees))
#V2=np.array([1/float(len(deltatees_II))]*len(deltatees_II))

plt.hist(deltatees,bins=binning,alpha=0.5,label="charge method 1",weights=V)

#plt.hist(deltatees_II,bins=binning,alpha=0.5,color='r',label="charge method 2",weights=V2)
plt.legend()

plt.show()



N=sum(np.asarray(deltatees)>=np.log10(6e-6))

print "livetime for this set of files: ",sum(livetime)," s"
