#!/usr/bin/env python

#-------------------------------------------------------------------------------
# Interactive line-based program to acquire data from a DPO3054 oscilloscope
#
# python        2.7         (http://www.python.org/)
# pyvisa        1.8         (http://pyvisa.sourceforge.net/)
# pyvisa-py     0.3.dev0    (https://github.com/hgrecco/pyvisa-py)
# python-tk                 (call: sudo apt-get install python-tk)
#
# to access an instrument via USB:
#----------------------------------
# pyUSB         1.0.0       (https://walac.github.io/pyusb/)
#
#
# Modified by Etienne bourbeau (etienne.bourbeau@icecube.wisc.edu)
# last modified: July 25th 2017
#
# NOTE: you must run this script as super-user to allow the oscilloscope to be
#       recognized through the USB port
#
# NOTE-II: If you want to save waveform in binary data, you might run into an
#          ascii encoding issue. To fix this, you have to replace 'ascii' by
#          'iso-88859-1' on line 323 of the pyvisa-py file sessions.py
#
# NOTE-III: the location of sessions.py depends on your installation. an
#           example location is /usr/local/lib/python2.7/dist-package/pyvisa-py/
#-------------------------------------------------------------------------------


import visa
import numpy as np
from struct import unpack
import pylab
import sys
import time
import matplotlib.pyplot as plt

import struct

from pyvisa import util


class scope_params:
    def __init__(self,scope):
        self.nacq=10
        self.nchannels=1
        self.triglvl=-100e-3
        self.trigsource="CH1"
        self.tracelength=2e-3
        self.tracescale=100
        self.savetype="bin"
        self.filename="test"

        # Display information:
        self.ymult = float(scope.query('WFMOutpre:ymult?').rstrip().split(" ")[1])
        self.yzero = float(scope.query('WFMOutpre:YZERO?').rstrip().split(" ")[1])
        self.yoffs = float(scope.query('WFMOutpre:YOFF?').rstrip().split(" ")[1])
        self.xincr = float(scope.query('WFMOutpre:xincr?').rstrip().split(" ")[1])

    
    def get_usr_inputs(self):
        self.filename=raw_input("Generic file name for this run? ")
        default=raw_input("Use default inputs (y/n)? ")
        n=0
        while (default!="n" and default!="y") and n<5:
            default=raw_input("Only type \"y\" or \"n\" : ")

            n+=1
            if n==4:
                print "you have problems...Try again with \"y\" or \"n\""
                sys.exit(-3)
                
        if default=="y":
            print "The default parameters that will be used are the following:"
            print "------------------------------------"
            print "# of acquisition:\t",self.nacq
            print "trigger level (V):\t",self.triglvl
            print "trigger source:\t\t",self.trigsource
            print "trace length (s):\t",self.tracelength
            print "savefile format:\t",self.savetype
            return
        else:
            self.nchannels=int(raw_input("Number of channels to acquire (1 to 4)? "))
            self.trigsource=raw_input("Trigger source (CH[1-4],AUX or LINE)? ")
            self.triglvl=float(raw_input("Trigger level (in V)? "))
            self.nacq = int(raw_input("number of traces to save? "))
            self.tracelength=float(raw_input("Duration of a trace (in s)? "))
            self.tracescale=float(raw_input("Scale of the acquisition window? (1/2/5)*(1/10/100)mV: "))
            self.savetype=raw_input("Savefile format (bin, asciii or csv)? ")
            if self.savetype!="bin":
                sys.exit("Ha! Only binary format is implemented right now!")
  

    def set_osc_inputs(self,scope):

        # Step1: verify that all source channels are opened

        # Step2: Set the trigger level
        scope.write("trigger:A:level:ch1 %f"%self.triglvl)
        # Step3: set the time correctly
        scope.write("horizontal:scale %f"%((self.tracelength/10.0)))
        scope.write("horizontal:position 5")
        
        # step4: Set the oscilloscope window correctly
        scope.write(":%s:scale %f"%(self.trigsource,self.tracescale/1000.0))

        # step5: set the inary encoding
        scope.write(":DATA:ENCdg SRPBINARY")
        scope.write(":WFMOutpre:BYT_NR 1")
        scope.write(":data:start 1")
        scope.write(":data:stop 5000000")


def load_oscilloscope(showstatus=True):
    rm= visa.ResourceManager('@py')
    print "Loading the available resource. Will show a\nlangid problem if you are not running as sudo...\n"
    address=rm.list_resources()[0]
    print "The oscilloscope is located at\n-----------------------------------\n",address
    print "-----------------------------------\n"

    scope = rm.open_resource(address)
    
    if showstatus:
        show_osc_status(scope)
    return scope

def show_osc_status(scope=None):

    if scope==None:
        print "ERROR: no instrument defined."
        sys.exit(-1)
    else:
        print "Querying Oscilloscope ID...\n--------------------------------------------"
        print scope.query("*IDN?"),"--------------------------------------------\n"
        time.sleep(5)
        print "Querying oscilloscope's data resolution..."
        print "------------------------------------------"

        print scope.query("CH1:SCALE?"),scope.query("CH1:YUNITS?"),\
            scope.query("CH1:COUpling?"),scope.query("CH1:TERMination?")

        print scope.query("CH2:SCALE?"),scope.query("CH2:YUNITS?"),\
            scope.query("CH2:COUpling?"),scope.query("CH2:TERMination?")

        print scope.query("CH3:SCALE?"),scope.query("CH3:YUNITS?"),\
            scope.query("CH3:COUpling?"),scope.query("CH3:TERMination?")
        
        print scope.query("CH4:SCALE?"),scope.query("CH4:YUNITS?"),\
            scope.query("CH4:COUpling?"),scope.query("CH4:TERMination?"),\
            "------------------------------------------\n"

        print "Querying display information..."
        print "-------------------------------"
        print "Vertical resolution:\t",float(scope.query('WFMOutpre:ymult?').rstrip().split(" ")[1])/1e-3,\
                                            " mV/sample"
        print "DC baseline:\t\t",scope.query('WFMOutpre:YZERO?').rstrip().split(" ")[1],\
                                            " V"
        print "Vertical offset:\t",scope.query('WFMOutpre:YOFF?').rstrip().split(" ")[1],\
                                            " samples"
        print "Time resolution:\t",float(scope.query('WFMOutpre:xincr?').rstrip().split(" ")[1])/1e-9," ns/sample"
        print "-------------------------------\n"
        
        
        print "Querying Trigger information..."
        print "----------------------------------"
        print scope.query("trigger:A:mode?").rstrip()
        trigtype=scope.query("TRIGger:A:type?")
        print trigtype.rstrip()
        #Additionnal information for edge-type trigger
        if trigtype.rstrip().split(" ")[1]=="EDGE":
            print scope.query("TRIGger:A:%s:slope?"%trigtype.split(" ")[1].rstrip()).rstrip()
            callstr="TRIGger:A:%s:source?"%trigtype.split(" ")[1].rstrip() # rstrip removes the newline character
            trigsource=scope.query(callstr).rstrip()
            print trigsource
            
            if trigsource.split(" ")[1]!="LINE":
                callstr="TRIGger:A:LEVEL:%s?"%trigsource.split(" ")[1]
                print scope.query(callstr)


def fillout_header(scope,headerfile):
    
    # Readout the parameter used in a binary-formatdata transfer
    # (as defined in page 58 of the programming guide)

    data_src=scope.query(":DATA:SOUrce?").rstrip()
    data_enc=scope.query(":DATA:ENCdg?").rstrip()
    data_beg=scope.query(":DATa:STARt?").rstrip()
    data_end=scope.query(":DATa:STOP?").rstrip()
    data_hor=scope.query(":HORizontal?").rstrip()
    triginfo=scope.query(":TRIGger:A?").rstrip()
    wfmo_enc=scope.query(":WFMOutpre:ENCdg?").rstrip()
    wfmo_bnr=scope.query(":WFMOutpre:BYT_Nr?").rstrip()
    wfmo_bor=scope.query(":WFMOutpre:BYT_Or?").rstrip()
    wfmo_fmt=scope.query(":WFMOutpre:BN_Fmt?").rstrip()
    wfmo_npt=scope.query(":WFMOutpre:NR_Pt?").rstrip()
    wfmo_wid=scope.query(":WFMOutpre:WFId?").rstrip()

    print "\nChecking the acquisition parameters:"
    print "------------------------------------------"
    print data_src
    print data_enc
    print data_beg
    print data_end
    print data_hor
    print triginfo
    print wfmo_enc
    print wfmo_bnr
    print wfmo_bor
    print wfmo_fmt
    print wfmo_npt
    print wfmo_wid
    print "------------------------------------------"

    
    ymult = float(scope.query('WFMOutpre:ymult?').rstrip().split(" ")[1])
    yzero = float(scope.query('WFMOutpre:YZERO?').rstrip().split(" ")[1])
    yoffs = float(scope.query('WFMOutpre:YOFF?').rstrip().split(" ")[1])
    xincr = float(scope.query('WFMOutpre:xincr?').rstrip().split(" ")[1])
    headerfile.write("Number of acquisition: %s\n"%params.nacq)
    headerfile.write("ymult = %.10f\n"%ymult)
    headerfile.write("yzero = %.10f\n"%yzero)
    headerfile.write("yoffs = %.10f\n"%yoffs)
    headerfile.write("xincr = %.10f\n"%xincr)
    headerfile.write("\nChecking the acquisition parameters:\n")
    headerfile.write("------------------------------------\n")
    headerfile.write(data_src+"\n")
    headerfile.write(data_enc+"\n")
    headerfile.write(data_beg+"\n")
    headerfile.write(data_end+"\n")
    headerfile.write(data_hor+"\n")
    headerfile.write(triginfo+"\n")
    headerfile.write(wfmo_enc+"\n")
    headerfile.write(wfmo_bnr+"\n")
    headerfile.write(wfmo_bor+"\n")
    headerfile.write(wfmo_fmt+"\n")
    headerfile.write(wfmo_npt+"\n")
    headerfile.write(wfmo_wid+"\n")

    

def osc_acquire(scope,params):
    time.sleep(5)
    scope.write("acquire:stopafter sequence") # puts the scope in single acquisition

    filename=params.filename
    
    # Fill out the headerfile
    headerfile=open(filename+'_header.txt',"w")
    fillout_header(scope,headerfile)
    headerfile.close()
    
    i=0
    n=0
    nfiles=1
    
    status=scope.query("BUSY?")
    runsplit=struct.pack('<i',-150) # this is a delimiter between runs in the bitstream
    
    while status.rstrip()!=":BUSY 0" and n<10:
        print "oscilloscope not ready, retrying in 1 s..."
        time.sleep(1)
        status=scope.query("BUSY?")
        n+=1
    if n==10:
        print "ERROR: the oscilloscope was too slow to respond."
        sys.exit(-1)

    # Option 1: convert data to ascii, then pickle the file 
    #data=[]

    # Option 2: Directly dump raw binary into file
    binfile=open(filename+"_0000.bin","wb")
    
    while i<=params.nacq:
        scope.write("acquire:state run")
        #data.append(scope.query_binary_values("wavfrm?",datatype='B',is_big_endian=False))
        scope.write("wavfrm?")
        data=scope.read_raw()
        binfile.write(data)
        binfile.write(runsplit)

        if (i+1)%1000==0:
            binfile.close()
            print "opening new file: ",filename+"_%04i.bin"%nfiles
            binfile=open(filename+"_%04i.bin"%nfiles,"wb")
            nfiles+=1
        
        i+=1
        while(scope.query("BUSY?").rstrip()==":BUSY 1"):
            time.sleep(0.01)


    #plot_binary_waveform(data)

def plot_binary_waveform(data):
    print len(data)
    offset, data_length = util.parse_ieee_block_header(data)
    expected_length = offset + data_length
    print "expected length: ",expected_length
    print "offset: ",offset
    print "data length: ",data_length
    F=struct.unpack('<%iB'%(data_length),data[offset+1:])
    print "first data point: ",F[0]
    plt.plot(F)
    plt.show()





    

if __name__=="__main__":

    print "********************************************"
    print " Hello! This is your oscilloscope interface."
    print "\n remember to run this program as sudo."
    print "********************************************"
    print "Let's see what we have here...\n"

    scope=load_oscilloscope()

    #Some lines to set the binary string decoding properly
    
    scope.encoding='latin-1'
    reload(sys)  # Reload does the trick!
    sys.setdefaultencoding('latin1')
    
    print "\nNow enter the parameters of the acquisition:"
    params=scope_params(scope)
    params.get_usr_inputs()
    params.set_osc_inputs(scope)
    print "thank you. Your acquisition will proceed in 5s..."
    osc_acquire(scope,params)
