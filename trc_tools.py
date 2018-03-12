#########################################
# Library of .trc file analysis code
#
#########################################

from pulsetools import *
from readTrc_master import readTrc

def parse_header_trc(trcformat):

    header_container=header_data()
    header_container.nacq=1
    
    header_container.windowscale['ymult']=float(trcformat['VERTICAL_GAIN'])
    header_container.windowscale['yoffs']=float(trcformat['VERTICAL_OFFSET'])
    header_container.windowscale['xincr']=float(trcformat['HORIZ_INTERVAL'])
    
    header_container.data['start']=int(trcformat['FIRST_POINT'])
    header_container.data['stop'] =int(trcformat['LAST_VALID_PNT'])

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
    header_container.time['duration']=header_container.data['stop']-header_container.data['start']+1
    header_container.time['vec']=np.arange(0,header_container.time['duration']*header_container.windowscale['xincr'],\
                                           header_container.windowscale['xincr'])
    header_container.triglvl=0.0
    
    return header_container



def load_data_trc(inputname,debug=False):

    seq_info=PMT_DAQ_sequence()
    X,Y,headerdata=readTrc.readTrc(inputname)

    header=parse_header_trc(headerdata)

    pedestal=compute_pedestal(Y)

    
    Y=-(Y-pedestal[0])
    charge,times=find_pulses_in_that_shit(header,Y,threshold=2e-3,Inverted=False,debug=debug)
    
    seq_info['charge']=charge
    seq_info['time']=times
    seq_info['livetime']=float(len(Y))*header.windowscale['xincr']
    seq_info['npulses']=len(charge)

    
    
    return [seq_info],header
