# SNOLAB Pulse acquisition tools

A set of scripts that were made to acquire waveform from several types of oscilloscopes, and analyze the data from these waveforms.

## Repository Layout

* `pulse_analyzer`:  Extract pulse information into a pickle file. This code has been set up to recognize the standard `*` wildcard, so you can specify either a specific file name, or simply the generic name of the run, and the program will automatically find all subfiles for that run. Note that you can easily delete/overwrite the pickle files produced, in case you are often modifying the `pulse_analyzer` code. DO NOT DELETE TRC FILES though!

* `pulse_II.py`: Generate charge and delta-t distribution plots. A set of default plots has been defined for various modes of acquisition (sequence, normal and flasher).

* `SNOLAB_produce_T-dependence_plots.py`: Wrapper script to automatically generate all relevant plots for the four DOMs tested at SNOLab. You can modify the plotting properties of each DOM in the file `utils/lab_doms.py`

* `VUVUZELA_produce_snolabified_plots.py`: Same sort of wrapper, but for data coming from vuvuzela simulation. The plotting options are a bit different, and can be modified in `utils/vuvuzeladoms.py`

*  `launch_pulse_analysis.py`: Sbatch submission script to automatically process trc data with `pulse_analyzer`, on hep01. You just need to specify a run number, and optionally a pulse width and threshold conditions for the pulse-finding algorithm.

* `produce_poissonian_data.py`: short script that produces a `pulse_II`-compatible pickle file containing noise data from an ideal, Poissonian source.

* `pulse_coincidence`: special script used to process the coincidence runs made at SNOLab. Takes in the name of the trigger and receiver channels, looks for coincident pulses in the traces, and compute a rate considering the livetime of each data frame.

* `snolabify_vuvuzela`: Takes in an i3 file and converts information of a pulse series into the PMT_daq_sequence format used in the SNOLab code.

* `snolabification`: Applies `snolabify_vuvuzela` to a folder of i3 files.



`oscilloscope_program.py` is meant to run a DPO3000 oscilloscope that's connected to your computer via USB. It saves waveforms in binary format

`trc_tool.py` is the library you need to read data saved by a LeCroy Waverunner 104Xi. The data acquisition process of this scope is programmed directly on the scope, which runs on Windows XP...

## Typical waveform acquisition (LeCroy Scope)

To analyze data from a `.trc` file:
