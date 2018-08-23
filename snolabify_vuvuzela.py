#!/usr/bin/env python


###############################################################
#   snolabify_vuvuzela.py
#
# Take in a vuvuzela file with a pulse series, and saves
# snolab-compatible files.
#
# Only do that for a couple of DOMs
#
###############################################################




def snolabify(input_file,output_file,pseries='I3MCPulseSeriesMap',dom='20-11'):

    import numpy as np

    # Retrieve the DOM id targeted
    #===================================================================
    om_to_check = int(dom.split('-')[1])
    str_to_check= int(dom.split('-')[0])


    # Define some containers
    #==================================================================
    list_of_seq = []
    

    # Open i3 file, loop over frames and save data fro the targeted dom
    #===================================================================
    
    from icecube import dataio
    from icecube import dataclasses,simclasses

    
    f = dataio.I3File(input_file)
    while f.more():

        frame = f.pop_daq()
        pseries = frame[pseries]
        times=[]
        charges=[]
        
        for DOM,hits in pseries:
            
            if DOM.string==str_to_check and DOM.om==om_to_check:
                for h in hits:
                    times.append(h.time)
                    charges.append(h.charge)


                    
        # Format the data to fit in a SNOLab-type file
        #=========================================================
        from pulsetools import PMT_DAQ_sequence


        seq_info=PMT_DAQ_sequence()
        
        N_pulses_in_series = len(times)
        times = np.array(times)
        charge = np.array(charges)

        seq_info['charge']=charge
        seq_info['time']=times*1.e-9
        seq_info['livetime']=50e-3
        seq_info['npulses']=N_pulses_in_series
        seq_info['mode']='normal'
    
    list_of_seq.append(seq_info)


    # Save the output file
    #==============================================================
    import pickle
    print "writing file..."
    pickle.dump(list_of_seq,open(output_file,"wb"))
    

    return 0


if __name__=='__main__':

    
    # Parse arguments
    #=========================================================
    
    import argparse
    parser = argparse.ArgumentParser('Convert a vuvuzela file into data usable on the SNOLab analysis code')

    parser.add_argument('--input-file',
                        help='input file name containing DOMLaunched vuvuzela data ',
                        default='vuvuzela_noiseonly_1min_01154.i3.bz2')
    
    parser.add_argument('--output-file',
                        help='output file with wavedeformed vuvuzela',
                        default="file_output.p")

    parser.add_argument('--pseries',
                        help='Pulse series to use.',
                        default="I3MCPulseSeriesMap")

    parser.add_argument('--dom',
                        help='Specify which dom to save (format: SS-DD)',
                        default = '20-11')

    args = parser.parse_args()
    

    

    # Launch the main code
    #===========================================================
    snolabify(args.input_file,
              args.output_file,
              args.pseries,
              args.dom)
