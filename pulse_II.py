#!/usr/bin/env python

#######################################################
# pulse II
# last update: July 22th 2018
#
# Author: Etienne Bourbeau
#         (etienne.bourbeau@icecube.wisc.edu)
#
# scripts that work on previously saved pickle file of
# an oscilloscope run
#
#######################################################

import sys
import numpy as np

# Define some functions
#============================================================


def burst_finder(sequence,burst_threshold):
    
    if sequence['npulses']>1:
        
        time = np.array(sequence['time'])
        charge = np.array(sequence['charge'])
        DT = time[1:]-time[:-1]

        burst_splitter = np.where((DT>burst_threshold)==1)[0]
        burst_times=np.split(time,burst_splitter+1)
        burst_charge=np.split(charge,burst_splitter+1)

        # remove the first and last bursts, as they are probably clipped

        burst_times=(burst_times[1:-1])
        burst_charge=burst_charge[1:-1]

        return burst_times,burst_charge
    
    else:

        return [np.array([0.0])],[np.array([0.0])]

def get_hist_stats(H,x,y):

    median = []
    avg = []
    
    for i in range(0,len(x)):

        histogram = H[i,:]
        data = []

        for j  in range(0,len(y)):
            data.append(np.repeat(y[j],histogram[j]))

        data = np.hstack(data)

        median.append(np.median(data))
        avg.append(np.mean(data))
        
    return median,avg



    
def parse_pseries(list_of_files,burst_thresh=1.e-6):

    import pickle
    from pulsetools import PMT_DAQ_sequence

    # Load data containers
    #------------------------------------------------------------------
    charge=[]
    Q_pair = []  # Summed charge of two consecutive pulses
    Q_ratio = [] # Ratio of Q(pulse 1) / Q(pulse 2)
    deltatees=[]
    npulses=[]
    livetime=[]
    mode=[]
    times=[]

    # Burst finder
    #------------------------------------------------------------------
    burst_threshold = 1.e-6 # Choosing a timescale much smaller than the thermal one
    bursts_charge_list = []
    bursts_time_list = []

    for pickled_file in list_of_files:

        if not pickled_file.endswith(".p"):
            sys.exit("ERROR: %s not a valid file name."%(pickled_file))
    
        if 'header' not in pickled_file:

            data = pickle.load(open(pickled_file,"rb"))
                

            for sequence in data:
                    
                livetime.append(sequence['livetime'])
                mode.append(sequence['mode'])

                bt,bc=burst_finder(sequence,burst_threshold)
                        
                if type(bc) is not list:
                    print type(bc)
                    sys.exit()
                            
                bursts_charge_list+=bc
                bursts_time_list+=bt
                        
                # Get rid of a nested list problem
                if isinstance(sequence,list):
                    sequence=sequence[0]

                if isinstance(sequence,PMT_DAQ_sequence):
                    
                    if sequence['npulses']>=1:
                                    
                        charge_array=np.array(sequence['charge'])                
                        kept = charge_array>burst_thresh
                        kept_charge = charge_array[kept]


                        if 'flasher' not in sequence['mode']:
                            
                            time_array  =np.array(sequence['time'])                    
                            kept_times  = time_array[kept] 
                            times.append(kept_times)
                                               
                            Q_pair.append(kept_charge[:-1]+kept_charge[1:])
                            Q_ratio.append(kept_charge[:-1]/kept_charge[1:])
                            deltatees.append(kept_times[1:]-kept_times[:-1])
                                                
                        npulses.append(sum(kept))
                        charge.append(kept_charge)
    
    return charge,times,deltatees,Q_pair,Q_ratio,livetime,npulses,mode,bursts_charge_list,bursts_time_list



# Run the code
#===============================================================================

if __name__=='__main__':

    import matplotlib
    from pulsetools import *
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt

    import pylab as plb
    from scipy.optimize import curve_fit
    from scipy import asarray as ar,exp
    from scipy.misc import factorial
    import scipy
    from matplotlib.colors import LogNorm
    from matplotlib.backends.backend_pdf import PdfPages

    parser = argparse.ArgumentParser(description="pulse II",
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument('-i', '--input',
                        dest='INFILE',
                        help="Input Data (pickle file)",
                        required=True)

    parser.add_argument('--input2',
                        default=None,
                        dest='INFILE2',
                        help="second set of data")

    parser.add_argument('--run',
                        dest='RUNID',
                        type=int,
                        help="number of the run")

    parser.add_argument('--threshold',dest='THRES',
                        type=float,
                        help="select only pulses above a certain threshold (in pC).",
                        default=-1000
    )

    parser.add_argument('--deadcut',dest='DCUT',
                        help="Apply the 6 us cut to compute the rate",
                        action='store_true'
    ) 

    parser.add_argument('--deadcutII',dest='DCUT2',
                        help="Apply the 2.45 us cut to compute the rate",
                        action='store_true'
    )


    parser.add_argument('--debug',dest='DEBUG',
                        action='store_true'
    )

    parser.add_argument('--dom',
                        default='',
                        help="DOM name")

    parser.add_argument('--temp',
                        type=int,
                        required=True,
                        help="Temperature")

    parser.add_argument('--scale',
                        type=float,
                        default=100,
                        help="Fixed scale of the log10dt plot")

    parser.add_argument('--output',
                        default="noise_pulse_analysis.pdf",
                        help="name of the output file containing all plots")

    parser.add_argument('--spe',
                        type=float,
                        required=True,
                        help="location of the SPE peak for the DOM")

    
    parser.add_argument('--qpair',
                        type=float,
                        required=True,
                        help="Scale of the Q pairs histograms")

    parser.add_argument('--show-plots',
                        action = "store_true")

    args = parser.parse_args()

    debug = args.DEBUG


    # building title for all plots
    #------------------------------------------------------------------
    titlename=args.dom+", -%i$^{\circ}$C"%args.temp


    # Load data containers
    #------------------------------------------------------------------

    charge=[]
    Q_pair = []  # Summed charge of two consecutive pulses
    Q_ratio = [] # Ratio of Q(pulse 1) / Q(pulse 2)
    deltatees=[]
    npulses=[]
    livetime=[]
    mode=[]
    times=[]


    # Burst finder
    #------------------------------------------------------------------

    burst_threshold = 1.e-6 # Choosing a timescale much smaller than the thermal one
    bursts_charge_list = []
    bursts_time_list = []


    # Load Data from the main input
    #------------------------------------------------------------------

    for pickled_file in glob.glob(args.INFILE):
    
        if 'header' not in pickled_file:

            data = pickle.load(open(pickled_file,"rb"))
                

            for sequence in data:
                    
                livetime.append(sequence['livetime'])
                mode.append(sequence['mode'])

                bt,bc=burst_finder(sequence,burst_threshold)

                if type(bc) is not list:
                    print type(bc)
                    sys.exit()
                            
                bursts_charge_list+=bc
                bursts_time_list+=bt
                        
                # Get rid of a nested list problem
                if isinstance(sequence,list):
                    
                    sequence=sequence[0]

                    if isinstance(sequence,PMT_DAQ_sequence):

                        
                        if sequence['npulses']>=1:
                                    
                            charge_array=np.array(sequence['charge'])
                            
                            kept = charge_array>args.THRES
                            kept_charge = charge_array[kept]


                            if 'flasher' not in sequence['mode']:
                                            
                                time_array  =np.array(sequence['time'])
                                
                                kept_times  = time_array[kept]
                                                
                                times.append(kept_times)

                                Q_pair.append(kept_charge[:-1]+kept_charge[1:])
                                Q_ratio.append(kept_charge[:-1]/kept_charge[1:])
                                deltatees.append(kept_times[1:]-kept_times[:-1])
                                                
                            npulses.append(sum(kept))
                            charge.append(kept_charge)

                              


    # Time-series quantities
    charge = np.concatenate(charge)
    Tiiime = np.concatenate(times)


    # Differential quantities
    time_deltas = np.concatenate(deltatees)
    qpairs = np.concatenate(Q_pair)
    qratio = np.concatenate(Q_ratio)

    # Livetime
    livetime=sum(livetime)
    rate = sum(npulses)/livetime
    print "Livetime: ",livetime," s"
    print "npulses :", sum(npulses)
    print "rate: ",rate," Hz"

    
    # Burst data
    bc_array  = bursts_charge_list # these are lists of arrays. One array = one burst
    bt_array  = bursts_time_list
    burst_sizes=[]
    burst_durations=[]
    burst_deltatees=[]
    for b in bt_array:

        if len(b)<2:
            bDT=[0.0]
        else:
            bDT = b[1:]-b[:-1]
        
        burst_deltatees.append(bDT)
    
        burst_sizes.append(len(b))
        burst_durations.append(sum(bDT))

    burst_deltatees =   np.array(burst_deltatees)
    burst_sizes = np.array(burst_sizes)
    burst_durations = np.array(burst_durations)
    print "Average burst size = ",sum(burst_sizes)/float(len(burst_sizes))






    #-----------------------------------------------------------------------------
    # Plotting begins
    #-----------------------------------------------------------------------------
    pdf = PdfPages(args.output)


    # 2D histogram of charge v. time of the second pulse w.r.t to its previous one
    #-----------------------------------------------------------------------------
    plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 22}

    matplotlib.rc('font', **font)

    xedges, yedges = np.arange(-8.0,-1.0,0.1), np.arange(-0,10,0.2)
    H, xedges, yedges = np.histogram2d(np.log10(time_deltas),qpairs/float(args.spe),(xedges,yedges))
    X, Y = np.meshgrid(xedges,yedges)

    # Get bin centers + add median of columns
    y_center = yedges[0:-1]+(yedges[1]-yedges[0])/2.0
    x_center = xedges[0:-1]+(xedges[1]-xedges[0])/2.0
    y_median,y_avg = get_hist_stats(H,x_center,y_center)
    plt.pcolormesh(np.transpose(X), np.transpose(Y), H/float(livetime),vmax=args.qpair)
    plt.plot(x_center,y_median,'k',linewidth=2.,label='median')
    plt.plot(x_center,y_avg,'c',linewidth=2.,label='avg')
    plt.plot(x_center,np.ones(len(x_center))*2.0,'w--',linewidth=3.0)
    plt.legend()

    plt.colorbar()
    plt.title(titlename)
    plt.xlabel('log10(dt)')
    plt.ylabel('charge of the pulse pair (pe)')
    pdf.savefig()
    if args.show_plots:
        plt.show()



    # 2D histogram: charge ratio v. delta-t
    #-----------------------------------------------------------------------------
    plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 22}

    xedges, yedges = np.arange(-8.0,-1.0,0.1), np.arange(-0,3.0,0.1)
    H, xedges, yedges = np.histogram2d(np.log10(time_deltas),qratio,(xedges,yedges))
    X, Y = np.meshgrid(xedges,yedges)


    # Get bin centers + add median of columns
    y_center = yedges[0:-1]+(yedges[1]-yedges[0])/2.0
    x_center = xedges[0:-1]+(xedges[1]-xedges[0])/2.0
    y_median,y_avg = get_hist_stats(H,x_center,y_center)

    plt.pcolormesh(np.transpose(X), np.transpose(Y), H/livetime,vmax=(args.qpair))
    plt.plot(x_center,y_median,'k',linewidth=2.,label='median')
    plt.plot(x_center,y_avg,'c',linewidth=2.,label='avg')
    plt.legend()

    plt.colorbar()
    plt.title(titlename)
    plt.xlabel('log10(dt)')
    plt.ylabel('$Q_{1}/Q_{2}$')
    pdf.savefig() 
    if args.show_plots:
        plt.show()



    # Plot the burst size distribution (number of pulses per bursts)
    #-----------------------------------------------------------------------------
    plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.hist(burst_sizes,bins=np.linspace(0.,15,16),color='g')
    plt.title(titlename)
    plt.xlabel("# of pulses per burst")
    plt.yscale('log')
    pdf.savefig() 
    if args.show_plots:
        plt.show()


    # Plot the burst durations (length of uninterrupted sequences of pulses
    #-----------------------------------------------------------------------------
    plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.hist(burst_durations/1.e-6,bins=np.linspace(0.,4,51))
    plt.title(titlename)
    plt.xlabel("Duration of bursts ($\mu$s)")
    plt.yscale('log')
    pdf.savefig() 
    if args.show_plots:
        plt.show()


    # Plot the burst duration profile
    #----------------------------------------------------------------------------
    plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    print min(burst_durations[burst_durations!=0])
    print max(burst_durations[burst_durations!=0])

    xedges, yedges = np.arange(2.0,20,1.0), np.arange(0,5,0.1)
    H, xedges, yedges = np.histogram2d(burst_sizes[burst_durations!=0],burst_durations[burst_durations!=0]/1.e-6,(xedges,yedges))
    X, Y = np.meshgrid(xedges,yedges)
    plt.pcolormesh(np.transpose(X), np.transpose(Y), H/float(livetime),norm=LogNorm(),vmax=1.e4)
    plt.colorbar()
    plt.title(titlename)
    plt.xlabel("# of pulses per burst")
    plt.ylabel('Duration of burst ($\mu$s)')
    pdf.savefig() 
    if args.show_plots:
        plt.show()





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


    def poisson(X,expected_rate):
    
        dt=[]    
        for i in range(0,int(livetime/0.05)):
            n = scipy.random.poisson(expected_rate*0.05)
            times = scipy.random.uniform(0.,0.05,size=n)
            x = np.sort(times)
            dt.append(x[1:]-x[0:-1])
                        
        dt = np.hstack(dt)
        y,_=np.histogram(dt,bins=X)
        y =np.concatenate([y,np.array([0.0])])
    
        return y



    def fit_uncorrelated_rate(poi_x,poi_y,livetime,bins):
                
        import scipy.optimize as optimization
        
        bestfitparam,cov= optimization.curve_fit(poisson, poi_x,poi_y,500,max_nfev=1000,method='trf',bounds=[100,6000],diff_step=1,verbose=2,xtol=1e-40)
        
        print "Best fit rate: ",bestfitparam[0]

        plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
        plt.title(titlename)
        plt.ylabel("count")
        plt.xlabel("delta-t (s)")
        plt.yscale('log')
                
        plt.hist(deltatees,bins=bins,color='b',label='run %04i'%args.RUNID,alpha=0.5)
                
        Yfit = poisson(X,bestfitparam[0])
        
        plt.plot(X, Yfit,'r',linewidth=3.0,label='Poisson Fit')
        plt.text(0.007,1000,'Poisson rate: %f Hz'%bestfitparam[0])
        plt.legend()
        pdf.savefig() 
        if args.show_plots:
            plt.show()
    

    # Plotting
    #-----------------------------------------------------------------------


    if not ('flasher' in mode):
    
        times = np.concatenate(times)
        deltatees = np.concatenate(deltatees)
        deltatees = deltatees[deltatees>0.]
        V=np.array([1/float(len(deltatees))]*len(deltatees))
    
        # Define the binning for delta-t histogram comparison
        binning = np.arange(-8.0,-1.0,0.1)
        binning_charge = np.arange(-5,20,0.2)


        # Remove sub-pe pulses from the array
        if debug:
            plt.plot(times)
            plt.show()
        

        
        # delta-t histogram
        #--------------------------------------------------------------------
        y,X=np.histogram(deltatees,bins=np.linspace(0.,0.01,201))

        y =  np.concatenate([y,np.array([0.0])])
        
        poi_x= X[(X>0.004)]
        poi_y= y[(X>0.004)]


        fit_uncorrelated_rate(poi_x,poi_y,livetime,X)



        
        # Charge distribution
        #----------------------------------------------------------------------------
        plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
        plt.ylabel("count")
        plt.xlabel("charge (pC)")
        plt.title(titlename)
        #plt.yscale('log')
        
        y,x,_=plt.hist(charge,bins=binning_charge,color='g',label='run %04i'%args.RUNID,alpha=0.5)
        plt.legend()
        pdf.savefig() 
        if args.show_plots:
            plt.show()
        
        dx = (x[1:]-x[0:-1])[0]
        x = x+dx/2

        
        # Log10(delta-t)
    
        Log10DT = np.log10(deltatees)
        low_dt = sum(Log10DT<=-7.)
        print "Fraction of hits below 10^-7 s: ",float(low_dt)/len(Log10DT)
    


        
        if args.DCUT:
            Log10DT = Log10DT[Log10DT>-5.221848]  # 6 us = -5.221848]
            V=np.array([1/float(len(Log10DT))]*len(Log10DT))
            rate = len(Log10DT+1)/livetime
            print "rate: %f Hz"%rate
        elif args.DCUT2:
            Log10DT = Log10DT[Log10DT>-5.610833915635467]
            V=np.array([1/float(len(Log10DT))]*len(Log10DT))
            rate = len(Log10DT+1)/livetime
            print "rate: %f Hz"%rate
                
        else:
            rate = sum(npulses)/livetime

        """
        with open("../analysis_data/Hitspool_2014_2017_dom05-05_example.p","rb") as hitspool:

        HS14,_=pickle.load(hitspool)
        HS14=np.asarray(HS14)
        #W=np.array([1/float(len(HS14))]*len(HS14))
        
        #W = np.array([sum(Log10DT[Log10DT>-6])/sum(HS14)]*len(HS14))
                
        #plt.hist(HS14,bins=binning,range=[-8,-1],histtype='step',linewidth=2.0,color='k',label="Hitspool 2014",weights=W)
        """
        

        plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
        plt.hist( Log10DT,bins=binning,alpha=0.5,label='run %04i'%args.RUNID,color='g',weights=np.ones(len(Log10DT))/float(livetime))
        plt.xlabel('log10($\Delta t$)')
        plt.ylabel('Rate (Hz)')
        plt.title(titlename)
        axes = plt.gca()
        axes.set_ylim([0,args.scale])
        plt.legend(loc='upper left')


        plt.text(-7,0.06,'Rate: %.3f Hz'%(rate))
        pdf.savefig() 
        if args.show_plots:
            plt.show()
    




        #================       Flasher run case ===================
        #-----------------------------------------------------------

    else:

        print "This is flasher data"
        binning_charge = np.arange(-1,300,5)
    
        # Charge distribution
        
        plt.ylabel("count")
        plt.xlabel("charge (pC)")
        #plt.yscale('log')
        y,x,_=plt.hist(charge,bins=100,color='g',alpha=0.5,label='flasher ON')

        if args.INFILE2 is not None:
            y2,x2,_=plt.hist(charge_II,bins=binning_charge,color='r',alpha=0.5,label='flasher OFF')
            plt.legend()
            plt.show()
            sys.exit()

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
        popt,pcov = curve_fit(gaussian,x,y,p0=[0000,-20,5])

        y2 = y-gaussian(x,*popt)
        if args.show_plots:
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
        

        if args.show_plots:
            plt.show()


    pdf.close()
