#!/usr/bin/env python
#########################################
# tools to read and format .vzp data
# (ie waveforms from IceCube HLC events)
#
# Retrieves the sequences of pulses from 
# a specific DOM, for a specific digitizer
# channel
#
# author: Etienne Bourbeau 
#        (etienne.bourbeau@icecube.wisc.edu)
#
#########################################


from collections import OrderedDict

# Default threshold level that will define
# what qualifies as a pulse in each type
# of waveforms
#
THRESHOLD = OrderedDict()
THRESHOLD['FADC'] = -0.15
THRESHOLD['ATWD'] = -0.3

def load_data_vzp(inputname, threshold=None, target_string=None, target_om=None, debug=False):

    import pickle
    import numpy as np
    from icecube.icetray import OMKey,I3Units
    from utils.vuvuzela_pulsetools import find_pulses_array
    
    #********** Raw data shows positive pulses ********
    nframes, alldoms_data = pickle.load(open(inputname))

    key = OMKey(target_string, target_om)

    if key not in alldoms_data.keys():
        print('ERROR: key {0} is not in the data recorded'.format(key))
        print('available keys are:')
        for k in sorted(alldoms_data.keys()):
            print(k)
        raise Exception
    
    good_data = alldoms_data[key]
    X = np.array(good_data['times'])*1.e-9 # convert into seconds
    Y = np.hstack(good_data['traces'])
    digitizer = good_data['type']

    
    file_deadtime = good_data['deadtime']

    D = {}
    D['VERT_COUPLING'] = 'DC_1_OHM'
    D['VERTUNIT'] = 'mV'
    D['HORUNIT'] = 's'
    if digitizer=='ATWD':
        D['HORIZ_INTERVAL'] = 3.3e-9
        nsamples = 128
        nsamp_for_pulse = 3
    elif digitizer=='FADC':
        D['HORIZ_INTERVAL'] = 25.0e-9
        nsamples = 256
        nsamp_for_pulse = 2
    else:
        sys.exit("ERROR: digitizer type unknown")


    # The data is stored as a list of continuous waveform hits
    # that does not take into account the number of frames that
    # were in the file. We have to manually assemble sequences
    # associated with a frame (and thus a livetime of 50ms)
    #
    # The time stamp resets after each frame, so we'll use that
    # to locate the proper split points.
    #
    # If (unlikely) there are less pslits than the recorded number
    # of frames, well just create empty sequences

    if nframes>1:
        splits = (X[1:]-X[:-1])<0.
        pivots = np.where(splits==True)[0]+1
        frame_times = np.split(X,pivots)
        npulses_in_frame = [len(x) for x in frame_times]

    else:
        frame_times = [X]
        npulses_in_frame = [len(X)]

        

        
    nstart=0
    all_sequences=[]
    for i in range(0,len(npulses_in_frame)):
        
        npulses = npulses_in_frame[i]
        t = frame_times[i]
        selected_trace = Y[nstart:(nstart+nsamples*npulses)]
        selected_times = np.tile(np.arange(0,nsamples)*D['HORIZ_INTERVAL'],len(t))+np.repeat(t,nsamples)
        
        
        charge, times = find_pulses_array(selected_times,-(selected_trace/I3Units.mV),D,
                                         sequence_time = selected_times,
                                         threshold=threshold,
                                         Nsample=nsamp_for_pulse,
                                         debug=debug)
        print(charge)

        seq_info=OrderedDict()
        seq_info['mode']='sequence'
        seq_info['charge']=charge
        seq_info['time']=times
        seq_info['livetime']=50.e-3
        seq_info['npulses']=len(charge)
        all_sequences.append(seq_info)
                
    return all_sequences,file_deadtime

        

        
if __name__=='__main__':

    
    # Parse arguments
    #=========================================================
    
    import argparse
    parser = argparse.ArgumentParser('Find pulses in vuvuzela waveforms.')

    parser.add_argument('-i', '--input-file',
                        help='input .vzp file name containing vuvuzela waveforms',
                        default='something_calibrated.vzp')
    
    parser.add_argument('-o', '--output-file',
                        help='output pickle file with the sequences',
                        default="some_sequence.p")

    parser.add_argument('--digitizer',
                        help='choose the digitizer type: ATWD or FADC',
                        default='ATWD')

    parser.add_argument('--tstring',type=int,help='target string',default=3)
    parser.add_argument('--tom',type=int,help='target OM',default=40)
    parser.add_argument('--debug',help='debug flag', action='store_true')

    args = parser.parse_args()

    print("processing file: ", args.input_file)
    
    sequences,file_deadtime = load_data_vzp(inputname = args.input_file,
                                            threshold = THRESHOLD[args.digitizer.upper()],
                                            target_string=args.tstring,
                                            target_om=args.tom,
                                            debug = args.debug)
        

    import pickle
    pickle.dump([sequences,file_deadtime],open(args.output_file,"wb"))

    print("*********PROGRAM COMPLETED**********")
