import h5py
from time import time
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# loading some utils
sys.path.append(os.environ["PWD"] + "/../")
import utilities as ut
# from utilities.analysis import rebin

create_calib = True
#run = "206162"
run = "206178"
#run = "206183"
DIR = "/media/sala/Elements/Data/Sacla/"

f = h5py.File(DIR + run + ".h5")


tag_list = f["/run_" + run + "/event_info/tag_number_list"][:]

# corr = None  # f_calib[run + "/MPCCD-1N0-M01-001/per_pixel_calib"][:]
if create_calib:
    f_calib = h5py.File(run + "_mpccd_calib.h5", "w")
    corr = ut.analysis.per_pixel_correction_sacla(f["/run_" + run + "/detector_2d_1/"], tag_list, thr=70)
    calib_dst = f_calib.create_dataset(run + "/MPCCD-1N0-M01-001/per_pixel_calib", data=corr)
else:
    f_calib = h5py.File(run + "_mpccd_calib.h5", "r")
    corr = f_calib[run + "/MPCCD-1N0-M01-001/per_pixel_calib"][:]

f_calib.close()
is_laser = f["/run_" + run + "/event_info/bl_3/lh_1/laser_pulse_selector_status"][:]

is_laser_off = is_laser == 0
is_laser_on = is_laser == 1
print is_laser_off.shape, is_laser_on.shape

init = time()
#roi = [[0, 1024], [325, 335]]  # X, Y
roi = []
[sum_image_on, spectrum_on], [sum_image_off, spectrum_off] = ut.cython_utils.get_spectrum_sacla(f["/run_" + run + "/detector_2d_1/"], tag_list, corr=corr, roi=roi, masks=[[is_laser_on], [is_laser_off]])
print sum_image_on, spectrum_on
print "time:", time() - init
spectrum_off = spectrum_off / spectrum_off.sum()
spectrum_on = spectrum_on / spectrum_on.sum()

plt.subplot(1, 2, 1)
plt.imshow(sum_image_on)
plt.subplot(1, 2, 2)
plt.plot(spectrum_off, label="laser off")
plt.plot(spectrum_on, label="laser on")
plt.plot(spectrum_on - spectrum_off, label="on - off", color="k")
plt.legend(loc="best")

plt.show()
