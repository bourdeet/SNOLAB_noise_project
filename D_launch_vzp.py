#!/usr/bin/env python

# Launch job scripts to produce SNOLAb-compatible
# vuvuzela with vzp_tools.py
#
# author: Etienne Bourbeau
#


if __name__=='__main__':


    import argparse
    import os
    import subprocess
    import time

    parser = argparse.ArgumentParser(description='Extract pulses from a pre-selected list of DOMs')

    parser.add_argument('--inputdir',
                        required=True,
                        help="Folder where the vzp files are located")

    parser.add_argument('--outputdir',
                        required=True,
                        help="Folder where the pickle files will be saved")

    parser.add_argument('--key',default=None,
                        help="write a key to locate a specific subset of files in the folder")

    parser.add_argument('--output-file',
                        help="name format of the output (no extension)",
                        default="run_vzp")

    parser.add_argument('--digitizer',
                            help='choose the digitizer type: ATWD or FADC',
                            default='ATWD')

    parser.add_argument('--threshold',
                        help='choose the pulse_detection threshold',
                        default=-0.5,type=float)

    parser.add_argument('--target',
                        help="name of the list containing target DOMs",
                        default="utils/target_original.py")

    args = parser.parse_args()


    ###########################
    import glob


    if not (args.key is None):
        files_to_process = sorted(glob.glob(args.inputdir+"/*"+args.key+"*.vzp"))
    else:
        files_to_process = sorted(glob.glob(args.inputdir+"/*.vzp"))

    bash_directory=args.inputdir+"/job_submit/"
        
    if not os.path.exists(bash_directory):
        print("creating new bash job folder...")
        os.makedirs(bash_directory)

    if not os.path.exists(args.outputdir):
        print('creating the output directory...')
        os.makedirs(args.outputdir)
        
    ############################

    #Hardcoded list of doms to harvest, this should be changed sometimes...
    exec("from utils.%s import targets"%(args.target.split('.')[-2].split('/')[-1]))

    for OM in targets:

        print("launching jobs for dom ",OM)
        i=0
        for f in files_to_process:

            output_file = args.outputdir+"/"+args.output_file+"_str%02i_om%02i_%05i.p"%(OM[0],OM[1],i)
            job_output  = bash_directory+"D_vzp_%02i_%02i_%05i_%s.out"%(OM[0],OM[1],i,args.digitizer)
            
            #print "\n"
            #print output_file
            #print "\n"

            submitfile=bash_directory+"vzp_%02i_%02i_%05i_%s.sh"%(OM[0],OM[1],i,args.digitizer)
            subfile= open(submitfile,"w")
            subfile.write("#!/bin/bash\n")
            subfile.write("#SBATCH --mem-per-cpu=5G\n")
            subfile.write("#SBATCH --output=%s\n"%job_output)
            
            Launch_code="python /lustre/hpc/icecube/bourdeet/SNOLAB/scripts/utils/vzp_tools.py "
            Launch_code+="--input-file %s "%(f)
            Launch_code+="--output-file %s "%(output_file)
            Launch_code+="--digitizer %s "%(args.digitizer)
            Launch_code+="--threshold %f "%(args.threshold)
            Launch_code+="--tstring %i "%(OM[0])
            Launch_code+="--tom %i "%(OM[1])

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
                    print("job %s has already completed successfully"%job_output)
                    continue
            

            executive_order="chmod +x %s"%(submitfile)
            subprocess.Popen(executive_order.split())
            launch_command="sbatch -p icecube %s "%(submitfile)
            print(launch_command+"\n")
            subprocess.Popen(launch_command.split())

            if i%20==0: time.sleep(0.5) # let the cluster breath in the jobs
            time.sleep(0.1)

        print("Done. What's next?")
        time.sleep(5.0)
