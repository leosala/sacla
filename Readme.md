# Overview
This repository holds all the scripts and ipython notebooks used for the PSI November SACLA beamtime ...

Before using go to utilities folder and execute `compile_cython.sh` to compile the cython code!

**Code is provided without any guarantee**

# Installation

## Python environment (Linux)

The suggested way to have a fully functional and updated Python environment is to use the Anaconda Python Distribution (https://store.continuum.io/cshop/anaconda/). Download it, install it in some local directory `<anaconda_dir>`, and then export the new `PATH`:

```
export PATH=<anaconda_dir>/bin:${PATH}'
```

When starting `ipython`, you should then get something like:

```
$ ipython
Python 2.7.9 |Continuum Analytics, Inc.| (default, Mar  9 2015, 16:20:48) 
Type "copyright", "credits" or "license" for more information.

IPython 2.2.0 -- An enhanced Interactive Python.
Anaconda is brought to you by Continuum Analytics.
Please check out: http://continuum.io/thanks and https://binstar.org
?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object', use 'object??' for extra details.

In [1]: 
```

## Getting this code

# Examples
## Adding DAQ information to DataConvert3 hdf5 files

`cd scripts && python convert_sacla_hdf5.py infile.h5 outfile.h5`

Please check the source code, as some parameters are hardcoded. Also, this script can be used to convert SACLA data files into having contiguous datasets (compared to one-image-per-tag datasets): please check the code.

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



