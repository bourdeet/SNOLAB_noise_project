#!/usr/bin/env python


import argparse
import fnmatch
import os
import subprocess
import time
import sys

parser = argparse.ArgumentParser(description='Create and launch data processing on HPC')
parser.add_argument('--run', dest="RUNID",help="run number", required=True)
parser.add_argument('--dir', dest="DIR",help="directory where the hdf5 files are located", required=True)
args = parser.parse_args()

#Change the name of the directory to one that suits your needs
output_directory="/groups/icecube/bourdeet/SNOLAB/March201888_data/run%04i/pickled/"%args.RUNID
bash_directory=output_directory+"/job_submit/"

for file in os.listdir(args.DIR):
    if fnmatch.fnmatch(file, 'Level3_TRUE__IC86.2012_corsika.0'+args.SIM+'.*.hd5'):

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
