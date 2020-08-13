#!/usr/bin/env python

#
# Convert wavecalibrated i3 files into vzp pickle files
# 
# These files are used to produce SNOLab-like plots with
# simulated vuvuzela data from in-situ DOMs
#
# author: Etienne Bourbeau (etienne.bourbeau@icecube.wisc.edu)


if __name__=='__main__':

        
    import argparse
    import os
    import subprocess
    import time

    parser = argparse.ArgumentParser(description='Produce vzp pickle files')

    parser.add_argument('--folder',
                        required=True,
                        help="Folder where the input i3 files are located")

    parser.add_argument('--key',default=None,
                        help="write a key to locate a specific subset of files in the folder")

    parser.add_argument('--digitizer',
                            help='choose the digitizer type: ATWD or FADC',
                            default='ATWD')

    parser.add_argument('--targets',
                        help="filename containing the list of target DOMs.",
                        default = "utils/target_original.py")

    args = parser.parse_args()


    ###########################
    import glob
    import sys

    if not (args.key is None):
        files_to_process = sorted(glob.glob(args.folder+"/*"+args.key+"*.i3*"))
    else:
        files_to_process = sorted(glob.glob(args.folder+"/*.i3*"))

        
    bash_directory=args.folder+"/job_submit/"
        
    if not os.path.exists(bash_directory):
        print "creating new bash job folder..."
        os.makedirs(bash_directory)

        
    ############################
    i=0
    for f in files_to_process:

        name = f.split("/")[-1].split(".")[0]
        

        output_file = args.folder+"/"+name+"_"+args.digitizer+".vzp"
        job_output  = bash_directory+"C_harvest_%s_%05i.out"%(args.digitizer,i)
        print "\n"
        print output_file
        print "\n"

        submitfile=bash_directory+"harv_%05i.sh"%(i)
        subfile= open(submitfile,"w")
        subfile.write("#!/bin/bash\n")
        subfile.write("#SBATCH --mem-per-cpu=2G\n")
        subfile.write("#SBATCH --output=%s\n"%job_output)
        
        Launch_code="python harvest_calibrated_waveform.py "
        Launch_code+="--input-file %s "%(f)
        Launch_code+="--output-file %s "%(output_file)
        Launch_code+="--digitizer %s "%(args.digitizer)
        Launch_code+="--targets %s "%(args.targets)

        Launch_code+="\n"
        subfile.write(Launch_code)
        subfile.close()
        i+=1

        # Check if the job output file already exists.
        # If it exists, see if it completed successfully

        if os.path.isfile(job_output):
            endline = ""
            for l in open(job_output): endline = l
            if "PROGRAM COMPLETED" in endline:
                print "job %s has already completed successfully"%job_output
                continue
            

        executive_order="chmod +x %s"%(submitfile)
        subprocess.Popen(executive_order.split())
        launch_command="sbatch -p icecube %s "%(submitfile)
        print launch_command+"\n"
        subprocess.Popen(launch_command.split())

        time.sleep(0.1)
