
#!/usr/bin/env python


##################################################################
#     SNOLAB_produce_T-dependence_plots_II.py
#
#  Plots the same things as version 1.0, but four all four SNOLab
#  DOMs on the same page, for a given temperature
#
###################################################################


import sys
sys.path.append("./utils/")


if __name__=='__main__':
    
    import matplotlib
    from pulsetools import *
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt

    from matplotlib.colors import LogNorm
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages

    parser = argparse.ArgumentParser(description="SNOLab plotter - 4 DOMs per page",
                                     formatter_class=RawTextHelpFormatter)
    
    parser.add_argument('--output',
                        default = "SNOLab_plots.pdf",
                        help="Name of the output pdf with plots"
    )
        
    parser.add_argument('--temperature',type=int,
                        default = 10,
                        help="Temperature of the DOM (positive integer)")

    parser.add_argument('--debug',dest='DEBUG',
                        action='store_true'
    )

    parser.add_argument('--show-plots',
                        help='show plots as they are generated',
                        action = "store_true")


    parser.add_argument('--highdt',
                        help='ctoff value for the high-dt computation.',
                        default='-4')
    
    args = parser.parse_args()
    debug = args.DEBUG



    print "Producing plots for SNOLab data..."

    
    # Load the plotting attributes of the IceCube doms
    #========================================================================
    from lab_doms import *
    from pulse_II import parse_pseries,get_hist_stats,fit_uncorrelated_rate
    from plotting_standards import *

    
    combined_results = {}


    # Loop over all DOMs and compute all relevant quantities
    #=========================================================================

    for dom in doms_to_plot:

        titlename="%s, -%i$^{\circ}$C"%(dom['name'],args.temperature)
        
        folder = "../analysis_data/March18/%s/m%i/pickled/"%(dom['name'],args.temperature)
        
        combined_results[dom['name']]= {}
        combined_results[dom['name']]['title'] = titlename
        pthreshold = dom['spe']*0.25
    
        # Call the pulse_series parser
        #====================================================================
        
        list_of_files = sorted(glob.glob(folder+"/*.p"))
        pserie_results = parse_pseries(list_of_files,pulse_threshold=pthreshold,burst_thresh=1.e-6)
    
        charge    = pserie_results[0]
        times     = pserie_results[1]
        deltatees = pserie_results[2]
        Q_pair    = pserie_results[3]
        Q_ratio   = pserie_results[4]
        livetime  = pserie_results[5]
        npulses   = pserie_results[6]
        mode      = pserie_results[7]
        bursts_charge_list = pserie_results[8]
        bursts_time_list = pserie_results[9]


        # Compute the physics quantities
        #=====================================================================
    
        # Time-series quantities
        charge = np.concatenate(charge)
        Tiiime = np.concatenate(times)


        # Differential quantities
        time_deltas = np.concatenate(deltatees)
        qpairs = np.concatenate(Q_pair)
        qratio = np.concatenate(Q_ratio)

        # Livetime
        combined_results[dom['name']]['livetime']=sum(livetime)
        rate = sum(npulses)/sum(livetime)

        #print "Livetime: ",livetime," s"
        #print "npulses :", sum(npulses)
        #print "rate: ",rate," Hz"
    
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

        # Log10DT plots
        selector = time_deltas>0.
        qpairs = qpairs[selector]
        qratio = qratio[selector]
        time_deltas = time_deltas[selector]
        
        Log10DT = np.log10(time_deltas)
        low_dt = sum(Log10DT<=-7.)
        print "Fraction of hits below 10^-7 s: ",float(low_dt)/len(Log10DT)
    


    
        #Store the relevant histogram in a dictionary for later plotting
        #============================================================================================


        H1,_,_ = np.histogram2d(Log10DT,qpairs/float(dom['spe']),(edges_dt,edges_Q))
        combined_results[dom['name']]['Qpairs'] = H1
        y_median,y_avg = get_hist_stats(H1,x_center,y_center)
        combined_results[dom['name']]['Qpairs_med'] = y_median
        combined_results[dom['name']]['Qpairs_avg'] = y_avg
    
    
        H2,_,_ = np.histogram2d(np.log10(time_deltas),qratio,(edges_dt,edges_Q))
        combined_results[dom['name']]['Qratio'] = H2
        y_median,y_avg = get_hist_stats(H2,x_center,y_center)
        combined_results[dom['name']]['Qratio_med'] = y_median
        combined_results[dom['name']]['Qratio_avg'] = y_avg


        #Burst 2D histograms
        H3,_,_ = np.histogram2d(burst_sizes[burst_durations!=0],burst_durations[burst_durations!=0]/1.e-6,(edges_ppb,edges_bl))
        combined_results[dom['name']]['burst2D'] = H3

        # Burst size
        H4,_ = np.histogram(burst_sizes,bins=edges_ppb)
        combined_results[dom['name']]['burst_size'] = burst_sizes

        # Burst durations
        H5,_ =np.histogram(burst_durations/1.e-6,bins=edges_bl)
        combined_results[dom['name']]['burst_duration'] = burst_durations/1.e-6

        # Raw deltatee
        H6,X6=np.histogram(time_deltas,bins=raw_dt_bins)
        combined_results[dom['name']]['delta_t'] = time_deltas
        combined_results[dom['name']]['delta_t_bins'] = X6
        # statistical uncertainties per bin
        sigma = np.ones(len(H6))
        for i in range(0,len(H6)):
            if H6[i]!=0:
                sigma[i]=1./np.sqrt(H6[i])

        # Take the center of the bins
        dx6 = (X6[1]-X6[0])/2.
        x_center6 = X6[:-1]+dx6

        # Select a subset of data to fit the thermal component
        combined_results[dom['name']]['poi_x']= x_center6[(x_center6>0.003)]
        combined_results[dom['name']]['poi_y']= H6[(x_center6>0.003)]
        combined_results[dom['name']]['sigma'] = sigma[(x_center6>0.003)]

        

        # Charge distribution
        H7,_=np.histogram(charge/float(dom['spe']),bins=binning_charge)
        combined_results[dom['name']]['charge'] = charge/float(dom['spe'])

        # Log 10 DT
        H8,_ = np.histogram(Log10DT,bins=binning_log10dt,weights=np.ones(len(Log10DT)))
        combined_results[dom['name']]['log10dt'] = Log10DT




    #==============================================================================
    #
    # Find out the low-DT segment with less than 1% deadtimes.
    #
    #==============================================================================
    
    for i in range(0,4):

        ID = doms_to_plot[i]['name']

        Dt   = np.sort(combined_results[ID]['log10dt'])
        livt = combined_results[ID]['livetime']

        lower_bound = np.log10(50.e-9)
        upper_bound = -6.2
        
        normal_data_start = float(args.highdt)
        normal_data_count = sum(Dt>=normal_data_start)
        
        count_below = sum(Dt<=upper_bound)-sum(Dt<=lower_bound)
        count_total = len(Dt[Dt>lower_bound])
        
        ratio_lowdt = count_below/float(count_total)
        
        error = 1./np.sqrt(count_below)
        
        print "\n############################"
        print "\n",ID,"\n"
        #print "count_below: ",count_below
        #print "total count: ",count_total
        print "count thermal: ",normal_data_count
        #print "livetime: ",livt
        #print "Ratio: ",ratio_lowdt
        #print "error: ",error
        #print "############################\n"
    

    #========================================================================================
    #-----------------------------------------------------------------------------
    # Plotting begins
    #-----------------------------------------------------------------------------
    #========================================================================================
    pdf = PdfPages(args.output)

    font = {'family' : 'serif',
            'weight' : 'bold',
            'size'   : 9}

    matplotlib.rc('font', **font)
    # Set up the main grid pf plots
    #=========================================================
    S = 5 # number of columns per plot
    n_x=2
    n_y=2


    # 2D histogram of charge v. time of the second pulse w.r.t to its previous one
    #-----------------------------------------------------------------------------

    # Set up the main grid pf plots
    #=========================================================

    gs = gridspec.GridSpec(n_x,n_x*S+1,wspace=2.0,hspace=0.4)
    f = plt.figure(figsize=(15,10))
    
    print "Creating Qpairs plots..."

    Ax = [None]*n_x*n_y

    for i in range(0,n_x*n_y):

        print i,',',i/n_x,(i%n_y*S),':',(i%2*S)+S
        
        Ax[i] = plt.subplot(gs[i/n_x,(i%n_y*S):(i%n_y*S)+S])

        ID = doms_to_plot[i]['name']
        qpair = doms_to_plot[i]['qpair']
        dom_data =  combined_results[ID]
        LVT = dom_data['livetime']
        
        pc = Ax[i].pcolormesh(np.transpose(X_dt), np.transpose(Y_Q), dom_data['Qpairs']/LVT,vmax=10)
        Ax[i].plot(x_center,dom_data['Qpairs_med'],'w',linewidth=2.,label='median')
        Ax[i].plot(x_center,dom_data['Qpairs_avg'],'c',linewidth=2.,label='avg')
        Ax[i].plot(x_center,np.ones(len(x_center))*2.0,'w--',linewidth=3.0)
        plt.legend()
    
        Ax[i].set_title(dom_data['title'])
        Ax[i].set_xlabel('log10(dt)')
        Ax[i].set_ylabel('charge of the pulse pair (pe)')
        Ax[i].yaxis.set_label_coords(-0.07,0.5)


    axes = plt.subplot(gs[:,n_y*S])
    plt.colorbar(pc, cax=axes)
    
    pdf.savefig()
    if args.show_plots:
        plt.show()


    # 2D histogram: charge ratio v. delta-t
    #======================================================================
    gs = gridspec.GridSpec(n_x,n_x*S+1,wspace=2.0,hspace=0.4)
    f = plt.figure(figsize=(15,10))

    print "Creating Qratio plots..."

    Ax = [None]*n_x*n_y
    for i in range(0,n_x*n_y):
        
        Ax[i] = plt.subplot(gs[i/n_x,(i%n_y*S):(i%n_y*S)+S])

        ID = doms_to_plot[i]['name']
        qpair = doms_to_plot[i]['qpair']
        dom_data =  combined_results[ID]
        LVT = dom_data['livetime']
    
        pc = Ax[i].pcolormesh(np.transpose(X_dt), np.transpose(Y_Q), dom_data['Qratio']/LVT,vmax=10)
        Ax[i].plot(x_center,dom_data['Qratio_med'],'w',linewidth=2.,label='median')
        Ax[i].plot(x_center,dom_data['Qratio_avg'],'c',linewidth=2.,label='avg')
        
        Ax[i].plot(x_center,np.ones(len(x_center)),'w--',linewidth=3.0)
        plt.legend()
    
        Ax[i].set_title(dom_data['title'])
        Ax[i].set_xlabel('log10(dt)')
        Ax[i].set_ylabel(r'charge ratio $Q_{1}/Q_{2}$')
        Ax[i].yaxis.set_label_coords(-0.07,0.5)


    axes = plt.subplot(gs[:,n_y*S])
    plt.colorbar(pc, cax=axes)
    
    pdf.savefig()
    if args.show_plots:
        plt.show()

    # Burst size distribution (number of pulses per bursts)
    #======================================================================
    S=5
    gs = gridspec.GridSpec(n_x,n_x*S+1,wspace=2.0,hspace=0.4)
    f = plt.figure(figsize=(15,10))

    print "Creating Burst properties plots..."

    Ax = [None]*n_x*n_y
    for i in range(0,n_x*n_y):

        Ax[i] = plt.subplot(gs[i/n_x,(i%n_y*S):(i%n_y*S)+S])

        ID = doms_to_plot[i]['name']
        dom_data =  combined_results[ID]
        LVT = dom_data['livetime']
        
        pc = Ax[i].pcolormesh(np.transpose(X_ppb), np.transpose(Y_bl),
                              dom_data['burst2D']/LVT,norm=LogNorm())
    
        Ax[i].set_title(dom_data['title'])
        Ax[i].set_xlabel(r'Duration of burst ($\mu s$)')
        Ax[i].set_ylabel('# of pulses per burst')
        Ax[i].yaxis.set_label_coords(-0.07,0.5)

    axes = plt.subplot(gs[:,n_y*S])
    plt.colorbar(pc, cax=axes)
        
    pdf.savefig()


    # 1D Burst length statistics
    #======================================================================
    S = 1 # number of columns per plot
    gs = gridspec.GridSpec(n_x,n_x*S,wspace=0.1,hspace=0.4)
    f = plt.figure(figsize=(15,10))

    print "Creating Burst length plots..."

    Ax = [None]*n_x*n_y
    for i in range(0,n_x*n_y):
        print i,',',i/n_x,(i%n_y*S),':',(i%2*S)+S
        Ax[i] = plt.subplot(gs[i/n_x,(i%n_y*S):(i%n_y*S)+S])

        ID = doms_to_plot[i]['name']
        dom_data =  combined_results[ID]
        LVT = dom_data['livetime']
        
        X = edges_ppb[0:-1]+(edges_ppb[1]-edges_ppb[0])/2.0

        Ax[i].hist(dom_data['burst_size'],
                   bins=edges_ppb,
                   weights = np.ones(len(dom_data['burst_size']))/LVT,
                   color='g')
    
        Ax[i].set_title(dom_data['title'])
        Ax[i].set_ylabel('Rate (Hz)')
        Ax[i].set_xlabel('# of pulses per burst')
        Ax[i].yaxis.set_label_coords(-0.07,0.5)
        Ax[i].set_yscale('log')
        
    pdf.savefig()

    
    # 1D Burst duration statistics
    #======================================================================
    S = 1 # number of columns per plot
    gs = gridspec.GridSpec(n_x,n_x*S,wspace=0.1,hspace=0.4)
    f = plt.figure(figsize=(15,10))

    print "Creating Burst duration plots..."

    Ax = [None]*n_x*n_y
    for i in range(0,n_x*n_y):
        print i,',',i/n_x,(i%n_y*S),':',(i%2*S)+S
        Ax[i] = plt.subplot(gs[i/n_x,(i%n_y*S):(i%n_y*S)+S])

        ID = doms_to_plot[i]['name']
        dom_data =  combined_results[ID]
        LVT = dom_data['livetime']
        
        pc = Ax[i].hist(dom_data['burst_duration'],
                        weights = np.ones(len(dom_data['burst_duration']))/LVT,
                        bins=edges_bl,
                        color='b')
    
        Ax[i].set_title(dom_data['title'])
        Ax[i].set_ylabel('Rate (Hz)')
        Ax[i].set_yscale('log')
        Ax[i].set_xlabel(r'Burst duration ($\mu s$)')
        Ax[i].yaxis.set_label_coords(-0.07,0.5)
        
    pdf.savefig()



    
    # Simple delta-t distribution with thermal rate fit
    #======================================================================
    S = 1 # number of columns per plot
    gs = gridspec.GridSpec(n_x,n_x*S,wspace=0.1,hspace=0.4)
    f = plt.figure(figsize=(15,10))

    print "Creating delta-t plots with thermal rate fit..."
    

    Ax = [None]*n_x*n_y
    for i in range(0,n_x*n_y):
        
        Ax[i] = plt.subplot(gs[i/n_x,(i%n_y*S):(i%n_y*S)+S])

        ID = doms_to_plot[i]['name']
        dom_data =  combined_results[ID]
        LVT = dom_data['livetime']

        poi_x = dom_data['poi_x']
        poi_y = dom_data['poi_y']/float(LVT)
        sigma = dom_data['sigma']/float(LVT)
        B = dom_data['delta_t_bins']

        Ax[i].hist(dom_data['delta_t'],bins=B,color='b',label='data',alpha=0.5,
                   weights = np.ones(len(dom_data['delta_t']))/float(LVT))
        
        Ax[i].errorbar(poi_x,poi_y,yerr=sigma,fmt='sk')
        Ax[i].set_title(dom_data['title'])
        Ax[i].set_ylabel("Rate(Hz)")
        Ax[i].set_xlabel("delta-t (s)")
        Ax[i].set_yscale('log')
        Ax[i].yaxis.set_label_coords(-0.07,0.5)

        
        Ax[i] = fit_uncorrelated_rate(poi_x,poi_y,sigma=sigma,ax=Ax[i])
        Ax[i].legend()
        
    pdf.savefig()
    

     # 1D charge distribution
    #======================================================================
    S = 1 # number of columns per plot
    gs = gridspec.GridSpec(n_x,n_x*S,wspace=0.1,hspace=0.4)
    f = plt.figure(figsize=(15,10))

    print "Creating charge duration plots..."

    Ax = [None]*n_x*n_y
    for i in range(0,n_x*n_y):
        
        Ax[i] = plt.subplot(gs[i/n_x,(i%n_y*S):(i%n_y*S)+S])

        ID = doms_to_plot[i]['name']
        dom_data =  combined_results[ID]
        LVT = dom_data['livetime']
        
        pc = Ax[i].hist(dom_data['charge'],bins=binning_charge,
                        weights=np.ones(len(dom_data['charge']))/LVT,
                        color='r',alpha=0.5)
    
        Ax[i].set_title(dom_data['title'])
        Ax[i].set_xlabel('charge (pe)')
        Ax[i].set_ylabel('Rate (Hz)')
        Ax[i].yaxis.set_label_coords(-0.07,0.5)
        
    pdf.savefig()
    
    
    # 1D Log10DT
    #======================================================================
    S = 1 # number of columns per plot
    gs = gridspec.GridSpec(n_x,n_x*S,wspace=0.1,hspace=0.4)
    f = plt.figure(figsize=(15,10))

    print "Creating Burst duration plots..."

    Ax = [None]*n_x*n_y
    for i in range(0,n_x*n_y):
        
        Ax[i] = plt.subplot(gs[i/n_x,(i%n_y*S):(i%n_y*S)+S])

        ID = doms_to_plot[i]['name']
        dom_data =  combined_results[ID]
        LVT = dom_data['livetime']
        
        pc = Ax[i].hist(dom_data['log10dt'],bins=binning_log10dt,
                        weights=np.ones(len(dom_data['log10dt']))/LVT,
                        color='g',alpha=0.5)
    
        Ax[i].set_title(dom_data['title'])
        Ax[i].set_xlabel('log10($\Delta t$)')
        Ax[i].set_ylabel('Rate (Hz)')
        Ax[i].yaxis.set_label_coords(-0.07,0.5)
        
    pdf.savefig()

    
    
    pdf.close()
    sys.exit()

    

