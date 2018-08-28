#!/usr/bin/env python

######################################################
#
# Special version of pulse_II that produces plots
# for sets of 9 doms at a time
#
# used with snolabified data from vuvuzela simulation
#######################################################

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

    parser = argparse.ArgumentParser(description="pulse III - plot vuvuzela quantities",
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument('--threshold',dest='THRES',
                        type=float,
                        help="select only pulses above a certain threshold (in pC).",
                        default=-1000
    )
    
    parser.add_argument('--output',
                        default = "VUVUZELA_plots.pdf",
                        help="Name of the output pdf with plots"
    )
    
    parser.add_argument('--debug',dest='DEBUG',
                        action='store_true'
    )

    parser.add_argument('--show-plots',
                        action = "store_true")


    args = parser.parse_args()
    debug = args.DEBUG



    print "Producing plots for vuvuzela simulation..."

    # Load the plotting attributes of the IceCube doms
    #========================================================================
    from vuvuzela_doms import *
    from pulse_II import parse_pseries,get_hist_stats


    # Define some elements that are common to several plots
    #========================================================================

    combined_results = {}
    edges_dt, edges_Q = np.arange(-8.0,-1.0,0.1), np.arange(-0,10,0.2)
    X_dt, Y_Q = np.meshgrid(edges_dt,edges_Q)
    x_center = edges_dt[0:-1]+(edges_dt[1]-edges_dt[0])/2.0
    y_center = edges_Q[0:-1]+(edges_Q[1]-edges_Q[0])/2.0
    
    edges_ppb, edges_bl = np.arange(2.0,20,1.0), np.arange(0,5,0.1)
    X_ppb, Y_bl = np.meshgrid(edges_ppb,edges_bl)


    # Loop over all DOMs and compute all relevant quantities
    #=========================================================================

    for dom in doms_to_plot:

        titlename="%s (%s), -%i$^{\circ}$C"%(dom['name'],dom['inice'],dom['T'])
        
        folder = "../analysis_data/March18/InIce-%s"%dom['inice']
        
        combined_results[dom['name']]= {}
        combined_results[dom['name']]['title'] = titlename
    
        # Call the pulse_series parser
        #====================================================================
        
        list_of_files = sorted(glob.glob(folder+"/*.p"))
        pserie_results = parse_pseries(list_of_files)
    
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
        H6,_=np.histogram(time_deltas,bins=np.linspace(0.,0.01,201))
        combined_results[dom['name']]['delta_t'] = time_deltas

        # Charge distribution
        binning_charge = np.arange(-5,20,0.2)
        H7,_=np.histogram(charge,bins=binning_charge)
        combined_results[dom['name']]['charge'] = charge

        # Log 10 DT
        binning = np.arange(-8.0,-1.0,0.1)
        H8,_ = np.histogram(Log10DT,bins=binning,weights=np.ones(len(Log10DT))/float(livetime))
        combined_results[dom['name']]['log10dt'] = Log10DT






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


    # 2D histogram of charge v. time of the second pulse w.r.t to its previous one
    #-----------------------------------------------------------------------------

    # Set up the main grid pf plots
    #=========================================================
    S = 5 # number of columns per plot
    gs = gridspec.GridSpec(3,3*S+1,wspace=2.0,hspace=0.4)
    f = plt.figure(figsize=(15,10))
    
    print "Creating Qpairs plots..."

    Ax = [None]*9
    print Ax
    for i in range(0,9):

        print i,',',i/3,(i%3*S),':',(i%3*S)+S
        
        Ax[i] = plt.subplot(gs[i/3,(i%3*S):(i%3*S)+S])

        ID = doms_to_plot[i]['name']
        qpair = doms_to_plot[i]['qpair']
        dom_data =  combined_results[ID]
    
        pc = Ax[i].pcolormesh(np.transpose(X_dt), np.transpose(Y_Q), dom_data['Qpairs']/float(livetime),vmax=5)
        Ax[i].plot(x_center,dom_data['Qpairs_med'],'k',linewidth=2.,label='median')
        Ax[i].plot(x_center,dom_data['Qpairs_avg'],'c',linewidth=2.,label='avg')
        Ax[i].plot(x_center,np.ones(len(x_center))*2.0,'w--',linewidth=3.0)
        plt.legend()
    
        Ax[i].set_title(dom_data['title'])
        Ax[i].set_xlabel('log10(dt)')
        Ax[i].set_ylabel('charge of the pulse pair (pe)')
        Ax[i].yaxis.set_label_coords(-0.07,0.5)


    axes = plt.subplot(gs[:,3*S])
    plt.colorbar(pc, cax=axes)
    
    pdf.savefig()
    if args.show_plots:
        plt.show()


    # 2D histogram: charge ratio v. delta-t
    #======================================================================
    S = 5 # number of columns per plot
    gs = gridspec.GridSpec(3,3*S+1,wspace=2.0,hspace=0.4)
    f = plt.figure(figsize=(15,10))

    print "Creating Qratio plots..."

    Ax = [None]*9
    for i in range(0,9):
        
        Ax[i] = plt.subplot(gs[i/3,(i%3*S):(i%3*S)+S])

        ID = doms_to_plot[i]['name']
        qpair = doms_to_plot[i]['qpair']
        dom_data =  combined_results[ID]
    
        pc = Ax[i].pcolormesh(np.transpose(X_dt), np.transpose(Y_Q), dom_data['Qratio']/float(livetime),vmax=5)
        Ax[i].plot(x_center,dom_data['Qratio_med'],'k',linewidth=2.,label='median')
        Ax[i].plot(x_center,dom_data['Qratio_avg'],'c',linewidth=2.,label='avg')
        Ax[i].plot(x_center,np.ones(len(x_center))*2.0,'w--',linewidth=3.0)
        plt.legend()
    
        Ax[i].set_title(dom_data['title'])
        Ax[i].set_xlabel('log10(dt)')
        Ax[i].set_ylabel(r'charge ratio $Q_{1}/Q_{2}$')
        Ax[i].yaxis.set_label_coords(-0.07,0.5)


    axes = plt.subplot(gs[:,3*S])
    plt.colorbar(pc, cax=axes)
    
    pdf.savefig()
    if args.show_plots:
        plt.show()

    # Burst size distribution (number of pulses per bursts)
    #======================================================================
    S = 5 # number of columns per plot
    gs = gridspec.GridSpec(3,3*S+1,wspace=2.0,hspace=0.4)
    f = plt.figure(figsize=(15,10))

    print "Creating Burst properties plots..."

    Ax = [None]*9
    for i in range(0,9):
        
        Ax[i] = plt.subplot(gs[i/3,(i%3*S):(i%3*S)+S])

        ID = doms_to_plot[i]['name']
        dom_data =  combined_results[ID]
    
        pc = Ax[i].pcolormesh(np.transpose(X_ppb), np.transpose(Y_bl), dom_data['burst2D']/float(livetime))
    
        Ax[i].set_title(dom_data['title'])
        Ax[i].set_xlabel(r'Duration of burst ($\mu s$)')
        Ax[i].set_ylabel('# of pulses per burst')
        Ax[i].yaxis.set_label_coords(-0.07,0.5)

    axes = plt.subplot(gs[:,3*S])
    plt.colorbar(pc, cax=axes)
        
    pdf.savefig()


    # 1D Burst length statistics
    #======================================================================
    S = 1 # number of columns per plot
    gs = gridspec.GridSpec(3,3*S,wspace=0.1,hspace=0.4)
    f = plt.figure(figsize=(15,10))

    print "Creating Burst length plots..."

    Ax = [None]*9
    for i in range(0,9):
        
        Ax[i] = plt.subplot(gs[i/3,(i%3*S):(i%3*S)+S])

        ID = doms_to_plot[i]['name']
        dom_data =  combined_results[ID]
        X = edges_ppb[0:-1]+(edges_ppb[1]-edges_ppb[0])/2.0


        Ax[i].hist(dom_data['burst_size'],
                   bins=edges_ppb,
                   color='g')
    
        Ax[i].set_title(dom_data['title'])
        Ax[i].set_ylabel('count')
        Ax[i].set_xlabel('# of pulses per burst')
        Ax[i].yaxis.set_label_coords(-0.07,0.5)
        
    pdf.savefig()

    
    # 1D Burst duration statistics
    #======================================================================
    S = 1 # number of columns per plot
    gs = gridspec.GridSpec(3,3*S,wspace=0.1,hspace=0.4)
    f = plt.figure(figsize=(15,10))

    print "Creating Burst duration plots..."

    Ax = [None]*9
    for i in range(0,9):
        
        Ax[i] = plt.subplot(gs[i/3,(i%3*S):(i%3*S)+S])

        ID = doms_to_plot[i]['name']
        dom_data =  combined_results[ID]
    
        pc = Ax[i].hist(dom_data['burst_duration'],
                        bins=edges_bl,
                        color='b')
    
        Ax[i].set_title(dom_data['title'])
        Ax[i].set_ylabel('count')
        Ax[i].set_yscale('log')
        Ax[i].set_xlabel(r'Burst duration ($\mu s$)')
        Ax[i].yaxis.set_label_coords(-0.07,0.5)
        
    pdf.savefig()
    
    
    # 1D Log10DT
    #======================================================================
    S = 1 # number of columns per plot
    gs = gridspec.GridSpec(3,3*S,wspace=0.1,hspace=0.4)
    f = plt.figure(figsize=(15,10))

    print "Creating Burst duration plots..."

    Ax = [None]*9
    for i in range(0,9):
        
        Ax[i] = plt.subplot(gs[i/3,(i%3*S):(i%3*S)+S])

        ID = doms_to_plot[i]['name']
        dom_data =  combined_results[ID]
    
        pc = Ax[i].hist(dom_data['log10dt'],bins=binning,
                        weights=np.ones(len(dom_data['log10dt']))/float(livetime),
                        color='g',alpha=0.5)
    
        Ax[i].set_title(dom_data['title'])
        Ax[i].set_xlabel('log10($\Delta t$)')
        Ax[i].set_ylabel('Rate (Hz)')
        Ax[i].yaxis.set_label_coords(-0.07,0.5)
        
    pdf.savefig()

    
    
    pdf.close()
    sys.exit()

    

