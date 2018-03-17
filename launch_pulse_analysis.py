#!/usr/bin/env python


import argparse
import fnmatch
import os
import subprocess
import time
import sys

parser = argparse.ArgumentParser(description='Create and launch data processing on HPC')

parser.add_argument('--run',
                    dest="RUNID",
                    type=int,
                    help="run number",
                    required=True)

args = parser.parse_args()

#Change the name of the directory to one that suits your needs

input_directory ="/groups/icecube/bourdeet/SNOLAB/March2018_data/run%04i/"%args.RUNID
output_directory=input_directory+"/pickled/"

bash_directory=output_directory+"/job_submit/"


if not os.path.exists(input_directory):
        sys.exit("ERROR: run data not found in %s"%input_directory)

else:

        if not os.path.exists(output_directory):
                os.makedirs(output_directory)

                bash_directory=output_directory+"/job_submit/"

        for trcfile in os.listdir(input_directory):
                if fnmatch.fnmatch(trcfile, '*run%04i*.trc'%(args.RUNID)):
                        print trcfile

                sys.exit()
                                   

                                   

        outhistogram= file[:-4]+".singlemuons.root"

        identity=file.split(".")[4]

        Launch_code="~/RIDE/histogram_production/RIDE_histogram_truth_barebones.py -i %s -o %s -s %i --hits"%(args.DIR+file,output_directory+outhistogram,int(args.SIM))

        #Add cuts to the processing or not:
	#Launch_code+=" --all-truth"


        submitfile=bash_directory+"histoprod_singlemuon_submit"+identity+".sh"
        
        with open(submitfile,"w") as subfile:
            subfile.write("#!/bin/bash\n")
            subfile.write(Launch_code)
 
        executive_order="chmod +x %s"%(submitfile)
        subprocess.Popen(executive_order.split())
        
        launch_command="sbatch -p icecube %s --mem-per-cpu=3G"%(submitfile)
        print launch_command
        subprocess.Popen(launch_command.split())
        
        time.sleep(0.1)
