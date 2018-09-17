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
    
    parser.add_argument('--output',
                        default = "VUVUZELA_plots.pdf",
                        help="Name of the output pdf with plots")
    
    parser.add_argument('--key',help="Special key to look for in the file names.",
                        default=None)
    
    parser.add_argument('--debug',dest='DEBUG',
                        action='store_true')

    parser.add_argument('--show-plots',
                        action = "store_true")

    parser.add_argument('--targets',
                        help="library of vuvuzela DOMs to compute",
                        default = "vuvuzela_doms")

    
    args = parser.parse_args()
    debug = args.DEBUG



    print "Producing plots for vuvuzela simulation..."

    # Load the plotting attributes of the IceCube doms
    #========================================================================
    exec("from %s import *"%(args.targets))
    from pulse_II import parse_pseries,get_hist_stats
    from plotting_standards import *
    
    combined_results={}
    data_name = args.output.split(".")[0]+".p"

    # Check if the data has already been acquired.
    #=========================================================================

    if not os.path.isfile(data_name):

        # Loop over all DOMs and compute all relevant quantities
        #=========================================================================

        for dom in doms_to_plot:
            
            titlename="%s (%s), -%i$^{\circ}$C"%(dom['name'],dom['inice'],dom['T'])
            print "\n",titlename
        
            folder = "../analysis_data/March18/target_doms_m30/InIce-%s"%dom['inice']
        
            combined_results[dom['name']]= {}
            combined_results[dom['name']]['title'] = titlename
            pthreshold = dom['spe']*0.25
            burst_thresh = 1.e-6
        
            # Call the pulse_series parser
            #====================================================================

            if args.key is None:
                list_of_files = sorted(glob.glob(folder+"/*.p"))
            else:
                list_of_files = sorted(glob.glob(folder+"/*"+args.key+"*.p"))
            
            print "Calling pserie parser with burst threshold of ",burst_thresh," and a pulse_threshold of ",pthreshold
            print "This will take a little time..."
            pserie_results = parse_pseries(list_of_files,
                                           pulse_threshold=pthreshold,
                                           burst_thresh=burst_thresh,
                                           withdeadtime = True)
    
            charge    = pserie_results[0]
            times     = pserie_results[1]
            deltatees = pserie_results[2]
            Q_pair    = pserie_results[3]
            Q_ratio   = pserie_results[4]
            livetime  = pserie_results[5]
            npulses   = pserie_results[6]
            mode               = pserie_results[7]
            bursts_charge_list = pserie_results[8]
            bursts_time_list   = pserie_results[9]
            deadtime           = pserie_results[10]
        
            print "...Done!"

            # Compute the physics quantities
            #=====================================================================
    
            # Time-series quantities
            charge = np.concatenate(charge)
            Tiiime = np.concatenate(times)
            Deads  = np.concatenate(deadtime)
            Deads = Deads[Deads>0.]*1.e-9 # Deadtimes converted in seconds

            # Differential quantities
            time_deltas = np.concatenate(deltatees)
            qpairs = np.concatenate(Q_pair)
            qratio = np.concatenate(Q_ratio)
            
            # Livetime
            combined_results[dom['name']]['livetime']=sum(livetime)-sum(Deads)
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
            
            print "median log10dt below -6: ",np.median(Log10DT[(Log10DT<-6.3)*(Log10DT>-7.)])

    
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
            H7,_=np.histogram(charge,bins=binning_charge)
            combined_results[dom['name']]['charge'] = charge/float(dom['spe'])

            # Log 10 DT
            H8,_ = np.histogram(Log10DT,bins=binning_log10dt,weights=np.ones(len(Log10DT)))
            combined_results[dom['name']]['log10dt'] = Log10DT
            combined_results[dom['name']]['deadtime'] = np.log10(Deads)

           
        pickle.dump(combined_results,open(data_name,"wb"))

    else:
        print "Retrieving data already saved..."
        combined_results = pickle.load(open(data_name))


    #==============================================================================
    #
    # Find out the low-DT segment with less than 1% deadtimes.
    #
    #==============================================================================
    
    for i in range(0,10):

        ID = doms_to_plot[i]['name']

        Dt   = np.sort(combined_results[ID]['log10dt'])
        dead = np.sort(combined_results[ID]['deadtime'])
        dead = dead[dead!=np.log10(50.e-9)]
        
        livt = combined_results[ID]['livetime']

        lower_bound = np.log10(50.e-9)
        upper_bound = -6.2

        normal_data_start = -4.0
        normal_data_count = sum(Dt>=normal_data_start)
        
        one_percent  = np.percentile(dead,1)
        ninety9_percent = np.percentile(dead,99)
        
        count_below = sum(Dt<=upper_bound)-sum(Dt<=lower_bound)
        count_total = len(Dt[Dt>lower_bound])

        deadcontam = sum(dead<=upper_bound)-sum(dead<=lower_bound)
        
        error = 1./np.sqrt(count_below)
        
        print "\n############################"
        print "\n",ID,"\n"
        print "count in the ROI: ",count_below
        print "total count: ",count_total
        print "count unaffected by deadtime: ",normal_data_count
        print "livetime: ",livt
        print "one_percent: ",one_percent
        print "99_percent: ",ninety9_percent
        print "statistical error: ",error
        print "deatime contamination: ",deadcontam
        print "total deadtime: ",len(dead[dead>lower_bound])
        print "############################\n"
    
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
    gs = gridspec.GridSpec(3,3*S+1,wspace=3.0,hspace=0.4)
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
        LVT = dom_data['livetime']
    
        pc = Ax[i].pcolormesh(np.transpose(X_dt), np.transpose(Y_Q), dom_data['Qpairs']/LVT,vmax=5)
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
        LVT = dom_data['livetime']
        
        pc = Ax[i].pcolormesh(np.transpose(X_dt), np.transpose(Y_Q), dom_data['Qratio']/LVT,vmax=5)
        Ax[i].plot(x_center,dom_data['Qratio_med'],'k',linewidth=2.,label='median')
        Ax[i].plot(x_center,dom_data['Qratio_avg'],'c',linewidth=2.,label='avg')
        Ax[i].plot(x_center,np.ones(len(x_center)),'w--',linewidth=3.0)
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
        LVT = dom_data['livetime']
        
        pc = Ax[i].pcolormesh(np.transpose(X_ppb), np.transpose(Y_bl),
                              dom_data['burst2D']/LVT,norm=LogNorm())
    
        Ax[i].set_title(dom_data['title'])
        Ax[i].set_ylabel(r'Duration of burst ($\mu s$)')
        Ax[i].set_xlabel('# of pulses per burst')
        Ax[i].yaxis.set_label_coords(-0.1,0.5)

    axes = plt.subplot(gs[:,3*S])
    plt.colorbar(pc, cax=axes)
        
    pdf.savefig()


    # 1D Burst length statistics
    #======================================================================
    S = 1 # number of columns per plot
    gs = gridspec.GridSpec(3,3*S,wspace=0.2,hspace=0.4)
    f = plt.figure(figsize=(15,10))

    print "Creating Burst length plots..."

    Ax = [None]*9
    for i in range(0,9):
        
        Ax[i] = plt.subplot(gs[i/3,(i%3*S):(i%3*S)+S])

        ID = doms_to_plot[i]['name']
        dom_data =  combined_results[ID]
        LVT = dom_data['livetime']
        
        X = edges_ppb[0:-1]+(edges_ppb[1]-edges_ppb[0])/2.0


        Ax[i].hist(dom_data['burst_size'],
                   weights = np.ones(len(dom_data['burst_size']))/LVT,
                   bins=edges_ppb,
                   color='g')
    
        Ax[i].set_title(dom_data['title'])
        Ax[i].set_ylabel('Rate (Hz)')
        Ax[i].set_xlabel('# of pulses per burst')
        Ax[i].yaxis.set_label_coords(-0.11,0.5)
        Ax[i].set_yscale('log')
        
    pdf.savefig()

    
    # 1D Burst duration statistics
    #======================================================================
    S = 1 # number of columns per plot
    gs = gridspec.GridSpec(3,3*S,wspace=0.2,hspace=0.4)
    f = plt.figure(figsize=(15,10))

    print "Creating Burst duration plots..."

    Ax = [None]*9
    for i in range(0,9):
        
        Ax[i] = plt.subplot(gs[i/3,(i%3*S):(i%3*S)+S])

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
        Ax[i].yaxis.set_label_coords(-0.11,0.5)
        
    pdf.savefig()

     # 1D charge distribution
    #======================================================================
    S = 1 # number of columns per plot
    gs = gridspec.GridSpec(3,3*S,wspace=0.2,hspace=0.4)
    f = plt.figure(figsize=(15,10))

    print "Creating charge distribution plots..."

    Ax = [None]*9
    for i in range(0,9):
        
        Ax[i] = plt.subplot(gs[i/3,(i%3*S):(i%3*S)+S])

        ID = doms_to_plot[i]['name']
        dom_data =  combined_results[ID]
        LVT = dom_data['livetime']
        
        pc = Ax[i].hist(dom_data['charge'],bins=binning_charge,
                        weights=np.ones(len(dom_data['charge']))/LVT,
                        color='r',alpha=0.5)
    
        Ax[i].set_title(dom_data['title'])
        Ax[i].set_xlabel('charge (pe)')
        Ax[i].set_ylabel('Rate (Hz)')
        Ax[i].yaxis.set_label_coords(-0.1,0.5)
        
    pdf.savefig()
    
    
    # 1D Log10DT
    #======================================================================
    S = 1 # number of columns per plot
    gs = gridspec.GridSpec(3,3*S,wspace=0.2,hspace=0.4)
    f = plt.figure(figsize=(15,10))

    print "Creating Log10DT plots..."

    Ax = [None]*9
    for i in range(0,9):
        
        Ax[i] = plt.subplot(gs[i/3,(i%3*S):(i%3*S)+S])

        ID = doms_to_plot[i]['name']
        dom_data =  combined_results[ID]
        LVT = dom_data['livetime']
        
        pc = Ax[i].hist(dom_data['log10dt'],bins=binning_log10dt,
                        weights=np.ones(len(dom_data['log10dt']))/LVT,
                        color='g',alpha=0.5)
        """
        # superimpose the deadtimes
        Y = dom_data['deadtime']
        Y1 = Y[Y!=np.log10(50.e-9)] #50e-9 for FADC, 6023 for ATWD
        Y2 = Y[Y==np.log10(50.e-9)]
        
        Ax2 = Ax[i].twinx()
        Ax2.hist(Y1,bins=binning_log10dt,
                 color='k',alpha=0.5)
        Ax2.get_yaxis().set_visible(False)

        Axb = Ax[i].twinx()
        Axb.plot([np.log10(6400.e-9),np.log10(6400.e-9)],#427 for ATWD, 6400 for FADC
                 [0.,1.],
                 color='r',linewidth=2.00)
        Axb.get_yaxis().set_visible(False)

        Ax3 = Ax[i].twinx()
        Ax3.hist(Y2,bins=binning_log10dt,
                 color='r',alpha=0.5)
        Ax3.get_yaxis().set_visible(False)
        """
        Ax[i].set_title(dom_data['title'])
        Ax[i].set_xlabel('log10($\Delta t$)')
        Ax[i].set_ylabel('Rate (Hz)')
        Ax[i].yaxis.set_label_coords(-0.07,0.5)

    pdf.savefig()

    
    
    pdf.close()
    sys.exit()

    

