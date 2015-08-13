# Useful scripts for data analysis at SACLA

**Warning**: some of these scripts are outdated, and will be changed soon... The scripts described in this README file should anyhow work with no issues.

## get_data
Gets data from the SACLA data buffer and converts them into HDF5 (via Maketaglist and DataConvert4)

### Usage

 * `./get_data.py <run>` - Get data for an individual run
 * `./get_data.py -l <run>` - Get data starting form the specified run until the latest run is reached
 * `./get_data.py -l -d <run>` -  - Get data starting form the specified run until the latest run is reached but keep waiting for more runs comming in

 * `./get_data.py -n <run>` - Get data for an individual run without the MPCCD detector data (will skip if the run had no MPCCD configured). Files are saved with an `_nompccd` suffix.


## get_roi_hdf5
Creates and HDF5 with ROI'd images. Warning: ROIs and DAQ quantities to be added are hardcoded at the beginning of: `sacla/scripts/get_roi_hdf5.py`

Example:
`python get_roi_hdf5.py -i /work/timbvd/hdf5 -o /work/leonardo/roi -d /work/timbvd/dark/dark_256635.h5`


# Other examples
## Adding DAQ information to DataConvert3 hdf5 files

`python convert_sacla_hdf5.py infile.h5 outfile.h5`

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
	
```
	python get_roi_hdf5.py -i /work/timbvd/hdf5/ -o /work/leonardo/roi/ -d /work/timbvd/dark/dark_256635.h5 256635
```

if no run number given, it starts an pyinotify daemon. WARNING! Start the script then from _the same node_ where DataConvert4 is running.
Please use always the last dark correction file (-d)! To connect to a specific node, use eg:
	 `qsub -I -l nodes=node22`


## Do some simple plots
	`cd scripts && python simple_analysis.py <filename.h5>`

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
