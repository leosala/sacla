# Useful scripts for data analysis at SACLA


# get_data
Gets data from the SACLA data buffer and converts them into HDF5 (via Maketaglist and DataConvert4)

## Usage

 * `./get_data.py <run>` - Get data for an individual run
 * `./get_data.py -l <run>` - Get data starting form the specified run until the latest run is reached
 * `./get_data.py -l -d <run>` -  - Get data starting form the specified run until the latest run is reached but keep waiting for more runs comming in

 * `./get_data.py -n <run>` - Get data for an individual run without the MPCCD detector data (will skip if the run had no MPCCD configured). Files are saved with an `_nompccd` suffix.


# `get_last_runs`

It gets the last run from 