#################################
#
# Customized binary data analysis
# library
#
#################################


from pulsetools import *
        
# See /home/bourdeet/.local/lib/python2.7/site-packages/pyvisa/util.py for inspiration

def try_readout(df,n):
    returnval=df.read(n)
    if len(returnval)<n:
        print ("WARNING: read less bytes than told")
        return []
    else:
        return returnval

def parse_binary_structure(fileptr):
    """Variation of the ieee parser 
            in pyvisa-py utils.py

    -reads the binary file one byte at a time
    -finds the '#' beginning
    -reads the header length
    -reads the data length
    -provides the total offset to the data
    """
    if fileptr==None:
        sys.exit("ERROR: File not found or badly read")
        
    offset_to_hash=0
    
    data=try_readout(fileptr,1)
    if len(data)>0:

        while data.find(b'#')==-1:
            offset_to_hash+=1
            data=try_readout(fileptr,1)

        head_length=int(try_readout(fileptr,1))
        data_length=try_readout(fileptr,head_length)
        offset=offset_to_hash+head_length
        try_readout(fileptr,1)

        return offset,int(data_length)

    else:
        print "WARNING: no more data to process"
        return None,None


def parse_header(headerfile,nfiles=1):
    
    header_container=header_data()
    
    with open(headerfile) as hf:
        for i, line in enumerate(hf):
            if i == 0:
                header_container.nacq=int(line.split(": ")[1])
            elif i ==1:
                header_container.windowscale['ymult']=float(line.split(" = ")[1])
            elif i==2:
                header_container.windowscale['yzero']=float(line.split(" = ")[1])
            elif i==3:
                header_container.windowscale['yoffs']=float(line.split(" = ")[1])
            elif i==4:
                header_container.windowscale['xincr']=float(line.split(" = ")[1])
            elif i==10:
                header_container.data['start']=int(line.split(" ")[1])
            elif i==11:
                header_container.data['stop'] =int(line.split(" ")[1])
            elif i==12:
                horizontal=line.split(";")
                header_container.time['scale']=float(horizontal[2].split(" ")[1])    # Time / div
                header_container.time['duration']=float(horizontal[3].split(" ")[1]) # Record length
                header_container.time['vec']=np.arange(0,header_container.time['duration']*header_container.windowscale['xincr'],\
                                                       header_container.windowscale['xincr'])

            elif i==13:
                trigstuff=line.split(";")
                header_container.triglvl=float(trigstuff[2].split(" ")[1])
                

        hf.close()

    print "Done. The parameters are:"
    print "------------------------------"
    print "nacq = ",header_container.nacq,"\n",\
        "window scale: ", header_container.windowscale,"\n",\
        "data stream: ",header_container.data,\
        "Trigger: ",header_container.triglvl,\
        "Time config: ", header_container.time
    
    return header_container




def load_data_bin(inputname,header,debug=False):
    
    print "loading file: ",inputname,"..."
    "parsing the binary file..."
    with open(inputname,"rb") as df:

              
        # Unpacking the binary data:
        #---------------------------
        # Data has been writtent according to the format
        # set-up in oscilloscope_program.py, that is:
        #
        # -> A little-endian format of bitstream ('<')
        # -> a ieee-compliant binary header (usually 291 bytes)
        # -> a set of N unsigned char ("B"), each 1 byte long
        # -> N is the number of points acquired per trace, and
        #    is given by the header information :WFMOUTPRE:NR_PT
        # -> Finally, a signed integer ('i', 4 bytes long)
        #    corresponding to a negative number (-150 usually).
        #    this is used to denote the end of a trace, and the
        #    beginning of another header
        #
        # -> The data is read one sequence at a time
        # -> The binary file is read once, in order
       
        all_the_info=[]
        all_the_info.append(header)
        seq_read=0
        
        offset,datalength=parse_binary_structure(df)
        sequence=try_readout(df,datalength+4)
        
        print "The initial offset is: ",offset
        print "This complete file set should contain ",header.nacq," traces."

    
        while sequence!='' and seq_read<(header.nacq):

            if seq_read%100==0:
                print seq_read," sequences read."

            data_from_this_seq=PMT_DAQ_sequence()
                    
            trace=struct.unpack("<%iBi"%(datalength),sequence)

            if debug:   
                plt.plot(trace)
                plt.xlabel("sample #")
                plt.ylabel("Signal (d.c.)")
                plt.show()
            
            y = ((np.asarray(trace[1:-2])-header.windowscale['yoffs'])*\
                 header.windowscale['ymult']+header.windowscale['yzero'])
            pedestal=compute_pedestal(y)
            data_from_this_seq['pedestal'] = pedestal

            if debug:
            
                plt.plot(header.time['vec'][:-2],y)
                plt.plot((header.time['vec'][0],header.time['vec'][-2]),(pedestal[0],pedestal[0]),'-r',linewidth=2)
                plt.plot((header.time['vec'][0],header.time['vec'][-2]),(pedestal[0]+3*pedestal[1],pedestal[0]+3*pedestal[1]),'-g',linewidth=2)
                plt.plot((header.time['vec'][0],header.time['vec'][-2]),(pedestal[0]-3*pedestal[1],pedestal[0]-3*pedestal[1]),'-g',linewidth=2)
            
                plt.xlabel("Time (s)")
                plt.ylabel("Signal (V)")
                plt.show()

                
            y=-(y-pedestal[0])
            charge,times=find_pulses_in_that_shit(header,y,threshold=-header.triglvl,Inverted=False,debug=debug)
            data_from_this_seq['charge']=charge
            data_from_this_seq['time']=times
            data_from_this_seq['livetime']=float(len(y))*header.windowscale['xincr']
            data_from_this_seq['npulses']=len(charge)

            #print data_from_this_seq
            all_the_info.append(data_from_this_seq)


            offset,datalength=parse_binary_structure(df)
            if datalength==None or offset==None:
                break
            sequence=try_readout(df,datalength+4)

            seq_read+=1
            del trace
            del y
            del charge
            del times
            del data_from_this_seq

        if seq_read !=(header.nacq-1):
            print "WARNING: This file contains less traces than expected."


    return all_the_info
