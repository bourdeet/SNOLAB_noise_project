#################################
# Some various filetype-specific
# libraries
#
################################


def load_data_pickle(inputname):
    header,time,data=pickle.load(open(inputname,"rb"))
    return header,time,data

def load_data_csv(inputname):
    
    print "detected a .csv file."
    
    header=[]
    time=[]
    data=[]
        
    i=0
    with open(inputname,"rb") as rawdata:
        headerrec=True  
        datarec=False
        while headerrec:
            i+=1
            line = rawdata.readline()
            header.append(line)
            split=line.split(",")
            if split[0]=='TIME':
                headerrec=False
                print "finished reading the header."
                datarec=True
        print "Moving on to the data acquisition..."
        while datarec:
            line=rawdata.readline()
            split=line.split(",")
            if len(split)>=2:
                x=split[0]
                y=split[1]
                time.append(float(x))
                data.append(-float(y))
            else:
                datarec=False

    print "Done.\n"
    print "Length of header:\t",len(header)
    print "Number of time samples:\t",len(time)
    print "Number of signal samples:\t",len(data)
    return header,time,data
