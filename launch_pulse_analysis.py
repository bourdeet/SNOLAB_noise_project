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


parser.add_argument('-o','--outname',
                    dest="OUTNAME",
                    help="prefix to the output file name",
                    default = "")

parser.add_argument('--width',
                    dest="PWIDTH",
                    help="pulse integration width",
                    type=int,
                    default = 10)


parser.add_argument('--thresh',
                    dest="THRESH",
                    help="threshold (default -500 microVolt",
                    type=float,
                    default = -0.0005)

args = parser.parse_args()


###########################
#--# of files per job
nfiles_per_job = 10
ntotal=0

#Change the name of the directory to one that suits your needs
input_directory ="/groups/icecube/bourdeet/SNOLAB/March2018_data/run%04i/"%args.RUNID
output_directory=input_directory+"/pickled/"

bash_directory=input_directory+"/job_submit/"

execution_directory = "/groups/icecube/bourdeet/SNOLAB/scripts/"


if not os.path.exists(input_directory):
        sys.exit("ERROR: run data not found in %s"%input_directory)

else:

        if not os.path.exists(output_directory):
                os.makedirs(output_directory)

                bash_directory=input_directory+"/job_submit/"

        for trcfile in os.listdir(input_directory):
                
                if fnmatch.fnmatch(trcfile, '*run%04i*.trc'%(args.RUNID)):

                        if ntotal%nfiles_per_job==0:
                                submitfile=bash_directory+"run%04i_submit_%i.sh"%(args.RUNID,ntotal/nfiles_per_job)
                                subfile= open(submitfile,"w")
                                subfile.write("#!/bin/bash\n")
                                subfile.write("#SBATCH mem-per-cpu=5G")

                        Launch_code=execution_directory+"pulse_analyzer.py -i %s -o %s/%srun%04i.p --pulse_width %i -t %f \n "%(input_directory+trcfile,output_directory,args.OUTNAME,args.RUNID,args.PWIDTH,args.THRESH)

                        subfile.write(Launch_code)
                        ntotal+=1
                        if ntotal%nfiles_per_job==0:
                                subfile.close()
                                executive_order="chmod +x %s"%(submitfile)
                                subprocess.Popen(executive_order.split())
                                launch_command="sbatch -p icecube %s  "%(submitfile)
                                print launch_command
                                subprocess.Popen(launch_command.split())
        
                                time.sleep(0.1)
