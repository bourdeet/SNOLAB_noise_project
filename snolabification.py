#!/usr/bin/env python


##################################################
#              wavedef_jobs.py
#
# runs process_to_wavedeform on a given folder
##################################################



if __name__=='__main__':



    # Get the name of the folder
    #======================================================
    
    import argparse
    
    parser = argparse.ArgumentParser("snolabify an entire folder!")
    parser.add_argument("--folder",
                        required = True,
                        help="Folder containing DOMlaunched i3 files")

    parser.add_argument('--pseries',
                        help='Pulse series to use.',
                        default="I3MCPulseSeriesMap")
    
    parser.add_argument('--dom',
                        help='dom for which you want a snolab-style pulse series',
                        default='20-11')
    
    args = parser.parse_args()


    # list files in the folder:
    #=======================================================

    import glob
    files_to_process = sorted(glob.glob(args.folder+"/*.i3.*"))

    # Produce an output directory for the dom you are analyzing
    #=======================================================

    import os
    
    new_directory = args.folder+"/"+args.dom+"/"
    
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    # Loop over file and run process_wavedeform
    #=======================================================

    import snolabify_vuvuzela as sno

    n_processed=0
    for f in files_to_process:

        if n_processed%100==0:
            print n_processed," processed files."
        
        if 'wavedeformed' not in f:


            # Retrieve the file number and define the output file name

            core_file = f.split(".i3")[0]
            file_number  = float(core_file.split('_')[-1])
            new_f = "snolabified_%s"%(args.dom)+f.split("/")[-1].split(".i3")[0]+".p"

            sno.snolabify(f,new_directory+new_f,args.pseries,args.dom)
            n_processed+=1

        
