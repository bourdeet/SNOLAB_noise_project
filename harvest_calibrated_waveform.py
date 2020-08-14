#!/usr/bin/env python


#######################################################################
#               harvest_calibrated_wafeforms.py
#
# Take in an i3 file containing data that's been processed up to
# WaveCalibrator. Extract the HLC waveforms from the desired
# digitizer and save them as sequences of arrays
#
#######################################################################




def harvest_calibrated_waveforms(inputfile, outputfile, digitizer, targets=None, debug=False):
    '''
    Inputs
    --------
    inputfile: str (name of an i3 file)

    outputfile: str (name of a vzp file)

    digitizer: str (type of digitization to use: ATWD or FADC)

    targets: str (name of a .py file containing a list of Tuples called target)

    debug: bool (debug flag)
    '''
    print("harvesting ",inputfile,"...")
    import numpy as np
    from icecube.icetray import OMKey,I3Units
    from icecube import dataclasses,simclasses
    from icecube import dataio
    import pickle

    if debug:
        import matplotlib.pyplot as plt


    exec("from utils.%s import targets"%(args.targets.split('.')[-2].split('/')[-1]))
    #convert tuples into OMKey objects
    targets_key=[]
    for e in targets:
        a = OMKey(e[0],e[1])
        targets_key.append(a)
        
    targets = targets_key

    
    F=dataio.I3File(inputfile)
    n_hits_total=0.
    n_hits_hlc=0.
    previous_atwd_1_hit = {}
    previous_hit = {}
    waveforms = {}
    for om in targets:
        
        waveforms[om] = {}  
        waveforms[om]['traces'] = []
        waveforms[om]['times'] = []
        waveforms[om]['type'] = digitizer
        waveforms[om]['n_HLC'] = 0.0
        waveforms[om]['n_tot'] = 0.0
        waveforms[om]['deadtime']= []
        previous_atwd_1_hit[om] = 0.0
        previous_hit[om]=None
        
    n_frames = 0
    while F.more():
        
        frame = F.pop_frame()

        for hits in frame['CalibratedWaveforms']:
            pmt = hits[0]
            if pmt in targets:
                for h in hits[1]:
                    if h.hlc:
                        waveforms[pmt]['n_HLC']+=1.
                    
                        if str(h.source)==digitizer:
                            #
                            # Store the waveform information
                            # Note: waveforms at this stage give positive pulses
                            #
                            waveforms[pmt]['traces'].append(h.waveform)
                            waveforms[pmt]['times'].append(h.time)
                            if debug:
                                plt.plot(np.array(h.waveform)/I3Units.mV)
                                plt.xlabel('sample #')
                                plt.ylabel('Waveform signal (mV)')
                                plt.show()


                            
                        # Deadtime calculations
                        #-----------------------------------------------------------------------
                        # Deal with the deadtime between ATWD recordings
                        # And overflow to higher amplification channels
                        #
                        # No matter the type of digitizer data wanted, both channels
                        # (ATWD and FADC) need to distinguish which ATWD fired on a hit
                        # before computing the associated deadtime
                        
                        
                        if digitizer=='ATWD':
                            if str(h.source)=='ATWD' and h.channel==0: # Deal with the deadtime between ATWD recordings
                                                                       # And overflow to higher amplification channels
                                # ATWD A
                                if h.source_index==0:
                                    previous_atwd_1_hit[pmt] = h.time
                                    waveforms[pmt]['deadtime'].append(6023.)
                                
                                elif h.source_index==1:
                                    
                                    if previous_atwd_1_hit[pmt] is None:
                                        print "at file ",inputfile," frame # ",n_frames
                                        sys.exit("ERROR: something is fishy: you found an atwd_b hit with no previous atwd_a hit.")
                                    
                                    tprevious = previous_atwd_1_hit[pmt]+32500 # time for ATWD A to finish digitizing
                                    dead_b = h.time+427 # start of the ATWD B deadtime
                                    waveforms[pmt]['deadtime'].append(tprevious-dead_b)
                                    previous_atwd_1_hit[pmt] = None


                        elif digitizer=='FADC' and str(h.source)=='FADC':

                            if previous_hit[pmt] is None:
                                print "at file ",inputfile," frame # ",n_frames
                                sys.exit("ERROR: something is fishy: First hit found was an FADC hit...")
                                
                            elif str(previous_hit[pmt].source)=='ATWD' and previous_hit[pmt].source_index==0: #if ATWD A:
                                previous_atwd_1_hit[pmt] = h.time
                                waveforms[pmt]['deadtime'].append(50) # only a 50ns delay before ATWD-b can kick in
                                
                            elif str(previous_hit[pmt].source)=='ATWD' and previous_hit[pmt].source_index==1: #if ATWD B

                                tprevious = previous_atwd_1_hit[pmt]+32500
                                waveforms[pmt]['deadtime'].append(tprevious-(h.time+6400))

                        previous_hit[pmt]=h

                            

                            
                    else:
                        print "HEY! found an SLC hit!!!"
                                    
                    waveforms[pmt]['n_tot']+=1.
                    
        n_frames+=1

    pickle.dump([n_frames,waveforms],open(outputfile,"wb"))
    return 

if __name__=='__main__':

    
    # Parse arguments
    #=========================================================
    
    import argparse
    parser = argparse.ArgumentParser('Harvest calibrated Waveforms')

    parser.add_argument('--input-file',
                        help='input file name containing calibrated waveforms',
                        default='something_calibrated.i3.bz2')
    
    parser.add_argument('--output-file',
                        help='output pickle file with the waveforms',
                        default="blabla_waveforms.vzp")

    parser.add_argument('--digitizer',
                        help='choose the digitizer type: ATWD or FADC',
                        default='ATWD')

    parser.add_argument('--targets',
                        help="filename containing the list of target DOMs.",
                        default = "utils/target_original.py")

    parser.add_argument('--debug',help='debug flag', action='store_true')
    args = parser.parse_args()


    harvest_calibrated_waveforms(inputfile  = args.input_file,
                                 outputfile = args.output_file,
                                 digitizer  = args.digitizer,
                                 targets    = args.targets,
                                 debug      = args.debug)


    print("*********PROGRAM COMPLETED**********")
