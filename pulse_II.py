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


import pylab as plb
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp



parser = argparse.ArgumentParser(description="pulse II",
                                 formatter_class=RawTextHelpFormatter)

parser.add_argument('-i', '--input',
                    dest='INFILE',
                    help="Input Data (pickle file)",
                    required=True)

parser.add_argument('--input2',
                    dest='INFILE2',
                    help="second set of data")

parser.add_argument('--run',
                    dest='RUNID',
                    type=int,
                    help="number of the run")

parser.add_argument('--threshold',dest='THRES',
                    type=float,
                    help="select only pulses above a certain threshold (in pC).",
                    default=0.0
                    )

parser.add_argument('--debug',dest='DEBUG',
                    action='store_true'
                    )

args = parser.parse_args()

debug = args.DEBUG

# Load data containers
#------------------------------------------------------------------

charge=[]
deltatees=[]
npulses=[]
livetime=[]
mode=[]
times=[]

# Load header information
#------------------------------------------------------------------

for pickled_file in glob.glob(args.INFILE):
    
        if 'header' not in pickled_file:

                data = pickle.load(open(pickled_file,"rb"))


                for sequence in data:
                        
                        mode.append(sequence['mode'])
                        # Get rid of a nested list problem
                        if isinstance(sequence,list):
                    
                                sequence=sequence[0]

                        if isinstance(sequence,PMT_DAQ_sequence):

                                livetime.append(sequence['livetime'])
                                
                                if sequence['npulses']>1:
                                        kept = sequence['charge']>args.THRES
                                        kept_charge = sequence['charge'][kept]
                                        kept_times  = sequence['time'][kept]
                                
                                
                                        npulses.append(sum(kept))
                                        charge.append(kept_charge)
                                        times.append(kept_times)
                                        deltatees.append(kept_times[1:]-kept_times[0:-1])

                              

charge = np.concatenate(charge)
deltatees = np.concatenate(deltatees)
times = np.concatenate(times)

# Defining fitting functions
def gaussian(x,A,mu,sigma):
        return A*1./(sigma)*np.exp(-((x-mu)**2.)/(2.*sigma**2.0))

def multigaus(x,A0,mu0,sigma0,A1,mu1,sigma1):

        npe=2
        pedestal = A0*(1./(sigma0*np.sqrt(np.pi))*exp(-((x-mu0)**2.)/(2.*sigma0**2.0)))
        signal   = A1*(1./(sigma1*np.sqrt(np.pi))*exp(-((x-mu1)**2.)/(2.*sigma1**2.0)))
        return pedestal+signal
                                

def SPE(x,mu_ped,s_ped,mu_exp,mu_1pe,s_1pe,n_pe_max=8):

        bkgd = gaussian(x,mu_ped,s_ped)*exp(-mu_exp)

        signal = 0.0
        for i in range(0,n_pe_max):
                signal+=gaussian(x,i*mu_1pe,np.sqrt(s_1pe))*poi(i,mu_exp)
        
        return signal+bkgd


# Plotting
#-----------------------------------------------------------------------


#================       Flasher run case ===================
#-----------------------------------------------------------

if 'flasher' in mode:

        print "This is flasher data"
        binning_charge = np.arange(-30,150,1.)
        
        # Charge distribution
        
        plt.ylabel("count")
        plt.xlabel("charge (pC)")
        #plt.yscale('log')
        y,x,_=plt.hist(charge,bins=binning_charge,color='g',alpha=0.5)

        dx = (x[1:]-x[0:-1])[0]
        x = (x+dx/2)[:-1]
        plt.plot(x,y,'x')

        # Getting the initial parameters for the SPE peak
        A0     = 40000.
        mu0    = -20.
        sigma0 = 10.
        A1     = 100.
        mu1    = 50.
        sigma1 = 25.
        


        #Peak fitting
        print "Fitting the pedestal..."
        popt,pcov = curve_fit(gaussian,x,y,p0=[40000,-20,5])

        y2 = y-gaussian(x,*popt)
        plt.show()
        plt.plot(x,y2)
        print popt
        print pcov

        print "\n Fitting the first pe peak..."
        
        popt,pcov = curve_fit(gaussian,x[50:],y2[50:],p0=[A1,mu1,sigma1])

        print "fitted."
        print "-------------------------------------------"
        print "Location of SPE peak: ",popt[1]
        
        plt.plot(x,gaussian(x,*popt),'ro:',label='fit')
        plt.xlabel('Charge (pC)')
        plt.ylabel('count')
        ax = plt.gca()
        plt.text(0.7, 0.8,'SPE location = %f'%popt[1], horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes, fontsize=12)

        
        #plt.plot(x,multigaus(x,Ao,npe_max,mu,sigma,mu0,sigma0,A0),'ro:',label='fit',linewidth=2.0)
        

        plt.show()
        sys.exit()

else:

        # Define the binning for delta-t histogram comparison
        binning = np.arange(-8.0,-1.0,0.1)
        binning_charge = np.arange(-5,50,1.)


        # Remove sub-pe pulses from the array
        if debug:
                plt.plot(times)
                plt.show()
        
        plt.ylabel("count")
        plt.xlabel("delta-t (s)")
        plt.yscale('log')
        y,x,_=plt.hist(deltatees,bins=500,color='r',label='run %04i'%args.RUNID,alpha=0.5)
        plt.legend()
        plt.show()

        
        # Charge distribution
        plt.ylabel("count")
        plt.xlabel("charge (pC)")
        plt.yscale('log')
        y,x,_=plt.hist(charge,bins=binning_charge,color='g',label='run %04i'%args.RUNID,alpha=0.5)
        plt.legend()
        plt.show()
        
        dx = (x[1:]-x[0:-1])[0]
        x = x+dx/2

        
        # Log10(delta-t)
        Log10DT = np.log10(deltatees)
        print len(Log10DT)
        
        with open("../analysis_data/Hitspool_2014_2017_dom05-05_example.p","rb") as hitspool:

                HS14,_=pickle.load(hitspool)
                HS14=np.asarray(HS14)
                W=np.array([1/float(len(HS14))]*len(HS14))
                plt.hist(HS14,bins=binning,range=[-8,-1],alpha=0.5,label="Hitspool 2014",weights=W)

    
        V=np.array([1/float(len(deltatees))]*len(deltatees))
        
        plt.hist( Log10DT,bins=binning,alpha=0.5,label='run %04i'%args.RUNID,weights=V)
        plt.xlabel('log10(delta-t)')
        plt.ylabel('normalized count')
        plt.legend(loc='upper left')

        rate = sum(npulses)/sum(livetime)
        
        print "livetime: ",sum(livetime)," s"
        print "npulses :", sum(npulses)
        print "rate: ",rate," Hz"
        
        plt.text(-7,0.06,'Rate: %.3f Hz'%(rate))
        plt.show()

        print "livetime: ",sum(livetime)," s"
        print "npulses :", sum(npulses)
        print "rate: ",sum(npulses)/sum(livetime)," Hz"
