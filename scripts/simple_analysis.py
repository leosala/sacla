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


fname = sys.argv[1]
is_roi = False  # is a ROI'd file?
#DIR = "/work/leonardo/roi/"
#DIR = "/work/timbvd/hdf5/"
thr = 70  # APU counting threshold

run = fname.split("/")[-1].replace("_roi", "").replace(".h5", "")

try:
    f = h5py.File(fname + "_roi.h5", "r")
    print f["/run_" + run + "/"].keys()
    is_roi = True
except:
    f = h5py.File(fname, "r")
    print f["/run_" + run + "/"].keys()

print run
print "/run_" + run + "/event_info/tag_number_list"
print f.keys()

print f["/run_" + run + "/"].keys()
tag_list = f["/run_" + run + "/event_info/tag_number_list"][:]
is_laser = f["/run_" + run + "/event_info/bl_3/lh_1/laser_pulse_selector_status"][:]

is_laser_off = is_laser == 0
is_laser_on = is_laser == 1

init = time()
roi = []
spectra_off = []
spectra_on = []
ax = []
fig = plt.figure(figsize=(10, 10))

for i in range(0, 2):
    [sum_image_on, spectrum_on], [sum_image_off, spectrum_off] = ut.analysis.get_spectrum_sacla(f["/run_" + run + "/detector_2d_" + str(i + 1)+ "/"], tag_list,  masks=[[is_laser_on], [is_laser_off]], thr=thr)

    if is_roi:
        image_sum = f["/run_" + run + "/detector_2d_" + str(i + 1) + "/image_avg"][:]

        ax.append(plt.subplot(2, 3, 1 + 3 * (i)))
        plt.imshow(image_sum)
        plt.colorbar()
        ax[-1].set_title("Image (mean)")

        ax.append(plt.subplot(2, 3, 2 + 3 * (i)))
        plt.imshow(sum_image_on + sum_image_off)
        ax[-1].set_title("Image (mean) ROI, thr > %s" % str(thr))
        spectra_off.append(spectrum_off / spectrum_off.sum())
        spectra_on.append(spectrum_on / spectrum_on.sum())

        ax.append(plt.subplot(2, 3, 3 + 3 * (i)))
        plt.plot(spectra_off[i], label="laser off")
        plt.plot(spectra_on[i], label="laser on")
        plt.plot(spectra_on[i] - spectra_off[i], label="on - off", color="k")
        ax[-1].set_title("Laser On - Laser Off, thr > %s" % str(thr))

    else:
        ax.append(plt.subplot(2, 2, 1 + 2 * (i)))
        plt.imshow(sum_image_on + sum_image_off)
        ax[-1].set_title("Image (mean) ROI, thr > %s" % str(thr))
        spectra_off.append(spectrum_off / spectrum_off.sum())
        spectra_on.append(spectrum_on / spectrum_on.sum())

        ax.append(plt.subplot(2, 2, 2 + 2 * (i)))
        plt.plot(spectra_off[i], label="laser off")
        plt.plot(spectra_on[i], label="laser on")
        plt.plot(spectra_on[i] - spectra_off[i], label="on - off", color="k")
        ax[-1].set_title("Laser On - Laser Off, thr > %s" % str(thr))
plt.suptitle("Up: MPCCD-1, Down: MPCCD-2")
plt.legend(loc="best")

plt.show()
