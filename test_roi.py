import h5py
import utilities as ut
from utilities.analysis import rebin
from time import time
import matplotlib.pyplot as plt
import numpy as np

run = "206162"
#run = "206178"
#run = "206183"
#DIR = "/media/sala/Elements/Data/Sacla/"
DIR = ""

f = h5py.File(DIR + run + ".h5")
f_out = h5py.File(DIR + run + "_roi.h5", "w")

grp = f_out.create_group("/run_" + run + "/detector_2d_1")
print grp.name
tag_list = f["/run_" + run + "/event_info/tag_number_list"][:]

#corr = f_calib[run + "/MPCCD-1N0-M01-001/per_pixel_calib"][:]

#is_laser = f["/run_" + run + "/event_info/bl_3/lh_1/laser_pulse_selector_status"][:]

init = time()
roi = [[0, 1024], [325, 335]]  # X, Y

ut.cython_utils.get_roi_dst(f["/run_" + run + "/detector_2d_1/"], f_out["/run_" + run + "/detector_2d_1/"], f_out, tag_list, roi=roi)
f_out.close()
print "time:", time() - init
