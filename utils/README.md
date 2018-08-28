# Utils

This is a repository containing a mixture of tools used by the progams in the main folder, along with old scripts that were used to harvest data from different oscilloscope (namely a DPO3000 oscilloscope)

## Currently used tools

* `trc_tool.py` is the library you need to read data saved by a LeCroy Waverunner 104Xi. The data acquisition process of this scope is programmed directly on the scope, which runs on Windows XP...

* `pulsetools` defines a couple of pulse-finding algorithms. One is reading a waveform linearly and looks for change in threshold, another does this more efficiently using numpy array, while a third one handles the special case of externally triggered flasher run.

* `lab_doms.py` defines the plotting properties of the SNOLab DOMs.

* `vuvuzela_doms.py` defines the plotting properties of the in-ice vuvuzela DOMs


## Old scripts 

* `oscilloscope_program.py` is meant to run a DPO3000 oscilloscope that's connected to your computer via USB. It saves waveforms in binary format

* `DPO3200bin_tools.py`: library used to handle binary data coming out of a Tektronix DPO3200 GHz oscilloscope.

* `oscilloscope_program.py`: waveform acquisition program that automatically saves waveforms to binary format (for the DPO3200 firmware)

* `otherfiles_tools.py`: library to readout other file formats used by oscilloscopes, such as csv data.

* `pulse_generator.py`: very basic code that produces fake pulses with a background pedestal