################################
# TO BE REWRITTEN!             #
################################

import h5py
from time import time
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

### loading some utils
TOOLS_DIR = "../../photon_tools"
# Loading ImagesProcessor
try:
    from photon_tools.images_processor import ImagesProcessor
    #from tools.plot_utilities import plot_utilities as pu
except:
    try:
        sys.path.append(TOOLS_DIR)
        from photon_tools.images_processor import ImagesProcessor
    except:
        print("[ERROR] cannot load ImagesProcessor library")
        exit()
        
# SACLA-specific tools     
SACLA_LIB = "../"
sys.path.append(SACLA_LIB)
import utilities as ut
# specific converters for the 2014-11 data taking. These should be customized per each beamtime!
from utilities import beamtime_converter_201411XX as sacla_converter


# Define SACLA quantities - they can change from beamtime to beamtime
daq_labels = {}
daq_labels["I0_down"] = "event_info/bl_3/eh_4/photodiode/photodiode_I0_lower_user_7_in_volt"
daq_labels["I0_up"] = "event_info/bl_3/eh_4/photodiode/photodiode_I0_upper_user_8_in_volt"
daq_labels["TFY"] = "event_info/bl_3/eh_4/photodiode/photodiode_sample_PD_user_9_in_volt"
daq_labels["photon_mono_energy"] = "event_info/bl_3/tc/mono_1_position_theta"
daq_labels["delay"] = "event_info/bl_3/eh_4/laser/delay_line_motor_29"
daq_labels["ND"] = "event_info/bl_3/eh_4/laser/nd_filter_motor_26"
daq_labels["photon_sase_energy"] = "event_info/bl_3/oh_2/photon_energy_in_eV"
daq_labels["x_status"] = "event_info/bl_3/eh_1/xfel_pulse_selector_status"
daq_labels["x_shut"] = "event_info/bl_3/shutter_1_open_valid_status"
daq_labels["laser_status"] = "event_info/bl_3/lh_1/laser_pulse_selector_status"
daq_labels["tags"] = "event_info/tag_number_list"

# run number
run = "257722"
DIR = "/home/sala/Work/Data/SACLA/"  # directory containing data
dark_fname = None  # eventual file containing dark corrections
fname = DIR + run + "_roi.h5"

# selections
sel = "(x_shut == 1) & (x_status == 1) & (I0_up > 0.01) & (I0_down > 0.01) & (ND > -1) & (photon_mono_energy) > 9"

# open data files
f = h5py.File(DIR + run + "_roi.h5")
if dark_fname is not None:
    f_calib = h5py.File(dark_fname, "r")
    corr = f_calib["/dark1"][:]
    f_calib.close()

# get DAQ quantities (only scalars)
df, filenames = ut.analysis.get_data_daq(fname, daq_labels, sacla_converter, t0=0, selection=sel)

# get laser on/off tags
is_laser_on_tags = df[df.is_laser == 1].index.tolist()
is_laser_off_tags = df[df.is_laser == 0].index.tolist()

# get spectra from Von Hamos, using laser on / off tags
#roi = [[0, 1024], [325, 335]]  # X, Y
ap = ImagesProcessor(facility="SACLA")
ap.add_analysis('get_projection', args={"axis": 1})
ap.add_analysis('get_mean_std')
ap.set_dataset('/run_%s/detector_2d_1' % run)
ap.add_preprocess("set_thr", args={"thr_low": 65})

# get the total spectra
results_on = ap.analyze_images(fname, tags=is_laser_on_tags)
spectrum_on = results_on["get_projection"]["spectra"].sum(axis=0)
results_off = ap.analyze_images(fname, tags=is_laser_off_tags)
spectrum_off = results_off["get_projection"]["spectra"].sum(axis=0)

spectrum_off = spectrum_off / spectrum_off.sum()
spectrum_on = spectrum_on / spectrum_on.sum()

# this is the average image from the Von Hamos
sum_image_on = results_on["get_mean_std"]["images_mean"]

# Plot!
plt.subplot(1, 2, 1)
plt.imshow(sum_image_on)
plt.subplot(1, 2, 2)
plt.plot(spectrum_off, label="laser off")
plt.plot(spectrum_on, label="laser on")
plt.plot(spectrum_on - spectrum_off, label="on - off", color="k")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
