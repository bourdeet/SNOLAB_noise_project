#########################################
# Library of .trc file analysis code
#
#########################################
import numpy as np

from pulsetools import find_pulses_in_that_shit,find_pulses_array,find_pulses_flasherrun
from pulsetools import PMT_DAQ_sequence
from pulsetools import header_data
from readTrc_master import readTrc
import matplotlib.pyplot as plt

def parse_header_trc(trcformat):

        header_container=header_data()
    
        header_container.nacq = trcformat['SUBARRAY_COUNT']
        header_container.impedance = float(trcformat['VERT_COUPLING'].split('_')[1])

        if trcformat['SUBARRAY_COUNT']>1:
                header_container.mode='sequence'
        else:
                header_container.mode='normal'
    
        header_container.windowscale['ymult']=float(trcformat['VERTICAL_GAIN'])
        header_container.windowscale['yoffs']=float(trcformat['VERTICAL_OFFSET'])
        header_container.windowscale['xincr']=float(trcformat['HORIZ_INTERVAL'])
        header_container.windowscale['xoffs']=float(trcformat['HORIZ_OFFSET'])
        header_container.triglvl=0.0

        timescale=trcformat['TIMEBASE']
        value=float(timescale.split("_")[0])
        units=timescale.split("_")[1][0:2]

        if units=="us":
                multiple=1e-6
        elif units=="ns":
                multiple=1e-9
        elif units=="ms":
                multiple=1e-3
        else:
                multiple=1.0

        header_container.time['scale']=value*multiple
        
    
        header_container.data['start']=int(trcformat['FIRST_POINT'])
        header_container.data['stop'] =int(trcformat['LAST_VALID_PNT'])
        
        header_container.time['scale']=value*multiple 
        header_container.time['duration']=header_container.data['stop']-header_container.data['start']+1
        header_container.time['vec']=np.arange(0,
                                               header_container.time['duration']*header_container.windowscale['xincr'],\
                                               header_container.windowscale['xincr'])


                
    
        return header_container


def load_data_trc(inputname,threshold,asSeq=False,debug=False):

        #********** Raw data show negative pulses ********

        seq_info=PMT_DAQ_sequence()
        X,Y,T,D=readTrc.readTrc(inputname)

        header=parse_header_trc(D)


        if header.mode=='normal':
        
                if debug:
                        plt.plot(X[0:10000],Y[0:10000])
                        plt.title('raw data')
                        plt.show()
                        
                #charge,times=find_pulses_in_that_shit(header,Y,threshold,Inverted=True,debug=debug)
                charge,times = find_pulses_array(X,Y,D,threshold=threshold,Nsample=3,debug=debug)
    
                seq_info['charge']=charge
                seq_info['time']=times
                seq_info['livetime']=float(len(Y))*header.windowscale['xincr']
                seq_info['npulses']=len(charge)
                seq_info['mode']='normal'

        

        elif header.mode=='sequence':

                trace_length = D['WAVE_ARRAY_COUNT']/D['SUBARRAY_COUNT']
                adjusted_time = (np.arange(0,len(X))%trace_length)*D['HORIZ_INTERVAL']
                trigtime_mapping = np.repeat(T['trigtime'],trace_length)  
                offset_mapping = np.repeat(T['offset'],trace_length)

                adjusted_time = adjusted_time+trigtime_mapping+offset_mapping
                
                if 'flasher' in inputname:
                        seq_info['mode']='flasher'
                        if asSeq:
                                charge = find_pulses_array(X,Y,D,
                                                           sequence_time=adjusted_time,
                                                           threshold=threshold,Nsample=3,debug=debug)
                        else:
                                charge = find_pulses_flasherrun(X,Y,D,debug=debug)
                                
                        times = None
                else:
                        seq_info['mode']='sequence'
                        charge,times = find_pulses_array(X,Y,D,sequence_time=adjusted_time,threshold=threshold,Nsample=3,debug=debug)

                seq_info['charge']=charge
                seq_info['time']=times
                seq_info['livetime']=adjusted_time[-1]-adjusted_time[0]
                seq_info['npulses']=D['SUBARRAY_COUNT']

        else:
                sys.exit('ERROR: acquisition mode unspecified')

                
        return [seq_info],header

        
