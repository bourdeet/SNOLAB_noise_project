# SNOLAB Pulse acquisition tools

A set of scripts that were made to acquire waveform from several types of oscilloscopes, and analyze the data from these waveforms.


`oscilloscope_program.py` is meant to run a DPO3000 oscilloscope that's connected to your computer via USB. It saves waveforms in binary format

`trc_tool.py` is the library you need to read data saved by a LeCroy Waverunner 104Xi. The data acquisition process of this scope is programmed directly on the scope, which runs on Windows XP...

## Typical waveform acquisition (LeCroy Scope)

To analyze data from a `.trc` file:

1. Use `pulse_analyzer.py` to extract pulse information into a pickle file. This code has been set up to recognize the standard `*` wildcard, so you can specify either a specific file name, or simply the generic name of the run, and the program will automatically find all subfiles for that run.

* Note that you can easily delete/overwrite the pickle files produced, in case you are often modifying the `pulse_analyzer` code. DO NOT DELETE TRC FILES though!

2. Use `pulse_II.py` To generate charge and delta-t distribution plots. A set of default plots has been defined for various modes of acquisition (sequence, normal and flasher).