#!/usr/bin/env python


##################################################################
#     SNOLAB_produce_T-dependence_plots.py
#
# -Loads the relevant data files associated with the SNOLab runs
#  performed on all four DOMs.
#
# -Computes the relevant quantities using pulse_II.py
#
# - Package all plots into a single PDF file
#
###################################################################



# Argument parser
# Should only be options for aestethics and ordering of plots in the PDF
#------------------------------------------------------------------
import argparse
import sys
sys.path.append("./utils/")
import glob
import subprocess

parser = argparse.ArgumentParser("Combine and plot all results from the SNOLab measurements.")

parser.add_argument('--rearange',
                    help='rearange the pdf pages',
                    default=True)

args = parser.parse_args()




from lab_doms import *



analysis_folder = "/home/etienne/NBI/SNOLab/analysis_data/March18/"
pulse_II = "/home/etienne/NBI/SNOLab/scripts/pulse_II.py"




for dom in doms_to_plot:

        for T in ['T10','T20','T30','T40']:

            print "\n"
            print dom['name'], T
            print "****************************\n"
        
            temp = int(T[1:])

            folder = analysis_folder+"/"+dom['name']+"/m"+T[1:]+"/pickled/"
            run_number = dom[T]
            name_pattern = "FINAL*run%04i_*.p"%(run_number)
        
            #Check that there is data in the requested folder
            #--------------------------------------------------
            list_of_available_files = sorted(glob.glob(folder+name_pattern))

            if len(list_of_available_files)==0:
                sys.exit("ERROR: No data available for dom %s at %s"%(dom['name'],T))
            else:
                print "running pulse_II..."
                #Launch pulse_II with the correct arguments
                #-------------------------------------------------
                arguments = " --input \"%s\" "%(folder+name_pattern)
                arguments+= "--run %i "%run_number
                arguments+= "--threshold %f "%(dom['spe']*0.25)
                arguments+= "--dom %s "%dom['name']
                arguments+= "--temp %i "%temp
                arguments+= "--scale %f "%(dom['scale'])
                arguments+= "--output %s_m%s.pdf "%(dom['name'],temp)
                arguments+= "--spe %f "%(dom['spe'])
                arguments+= "--qpair %f "%(dom['qpair'])
            
                command = pulse_II+arguments

                #print command
                subprocess.Popen(command,shell=True).wait()

            


if args.rearange:
    # Load all PDF and use pdftk to remerge them
    toplot = {1:'Q_per_pulse_pair',
              2:'Q_ratio',
              3:'pulse_per_burst',
              4:'duration_per_burst',
              5:'size_v_duration',
              6:'deltaT',
              7:'SPE',
              8:'Log10DT'}

    for name in doms_to_plot:
        n = name['name']
    
        for p in toplot:
            command = ["pdftk A={0}_m10.pdf B={0}_m20.pdf C={0}_m30.pdf D={0}_m40.pdf cat A{1} B{1} C{1} D{1} output {0}_{2}.pdf".format(n,p,toplot[p])]
            print command
            subprocess.Popen(command,shell=True).wait()
