# Overview
This repository holds all the scripts and ipython notebooks used for the November SACLA beamtime ...

Before using go to utilities folder and execute `compile_cython.sh` to compile the cython code!

# Examples
## Running the DataConvert4 automatic converter:
`./get_data.py -l -d <runnumber>`

e.g.
`./get_data.py -l -d 256635`

Best usage: 
+ for normal conversion:
	`./get_data.py -l -d -j 3 256635`
+ without MPCCD
	`./get_data.py -n -l -d -j 3 256635`


## Getting information on scans
Edit check_scans.sh (initial and final date, add new quantities), then just:
	`./check_scans.sh`


## Create ROI'd files
	cd sacla/scripts && python get_roi_hdf5.py -i /work/timbvd/hdf5/ -o /work/leonardo/roi/ -d /work/timbvd/dark/dark_256635.h5 256635

if no run number given, it starts an pyinotify daemon. WARNING! Start the script then from _the same node_ where DataConvert4 is running
please use always the last dark correction file (-d)! To connect to a specific node, use eg:
	 `qsub -I -l nodes=node22`



## Do some simple plots
	`cd sacla/scripts && python simple_analysis.py <filename.h5>`

it will plots:
- if on a normal file, average of threholded images, plus laser on - laser off spectra
- if on a ROI'd file, as above, plus original averaged dark subtracted image

You can also use the `get_pump_probe.py`, which is an improved multiprocessing version.

## Plot monodimensional quantities vs scans

Plot only one scan, e.g.: 
`cd scripts && python plot_scan.py --start_run <run> --end_run <run> --check`

Compare various scans, e.g.: 
`cd scripts && python compare_scan.py`

Usually these scripts need some tweaking (data filtering, input file names, etc), so please have a look at the code. 



