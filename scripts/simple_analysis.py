import h5py
from time import time
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from math import asin, sin, pi

# loading some utils
sys.path.append(os.environ["PWD"] + "/../")
import utilities as ut
# from utilities.analysis import rebin

IOlow = "/event_info/bl_3/eh_4/photodiode/photodiode_I0_lower_user_7_in_volt"
IOup = "/event_info/bl_3/eh_4/photodiode/photodiode_I0_upper_user_8_in_volt"
PDSample = "/event_info/bl_3/eh_4/photodiode/photodiode_sample_PD_user_9_in_volt"
Mono = "/event_info/bl_3/tc/mono_1_position_theta"
Delay = "/event_info/bl_3/eh_4/laser/delay_line_motor_29"
ND = "/event_info/bl_3/eh_4/laser/nd_filter_motor_26"


def get_energy_from_theta(theta_position):
    # Todo: Most probably these variables need to be read out from the control system ...
    try:
        theta_position = float(theta_position.replace("pulse", ""))
    except:
        theta_position = float(theta_position)

    theta_coeff = 25000.  # in [pulse/mm]
    lSinbar = 275.0  # in [mm]
    theta_offset = -15.0053431  # in [degree]
    dd = 6.270832

    theta = asin((theta_position / theta_coeff) / lSinbar + sin(theta_offset * pi / 180.0)) * 180.0 / pi - theta_offset
    energy = (12.3984 / (dd)) / sin(theta * pi / 180.0)

    return energy  # , units


def get_delay_from_pulse(pulse, t0=0):
    """"""
    magic_factor = 6.66 / 1000.
    return (float(pulse) * magic_factor) - t0


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
    is_roi = False

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

photon_energy = f["/run_" + run + "/event_info/bl_3/oh_2/photon_energy_in_eV"][:]
is_xray = f["/run_" + run + "/event_info/bl_3/eh_1/xfel_pulse_selector_status"][:]
is_laser = f["/run_" + run + "/event_info/bl_3/lh_1/laser_pulse_selector_status"][:]
tags = f["/run_" + run + "/event_info/tag_number_list"][:]

is_laser_off = is_laser == 0
is_laser_on = is_laser == 1
t_conv = np.vectorize(get_energy_from_theta)
d_conv = np.vectorize(get_delay_from_pulse)

iol = np.array(f["/run_" + run + IOlow][:])
iou = np.array(f["/run_" + run + IOup][:])
spd = np.array(f["/run_" + run + PDSample][:])
mono = t_conv(np.array(f["/run_" + run + Mono][:]))
nd = np.array(f["/run_" + run + ND])
delay = np.array(f["/run_" + run + "/event_info/bl_3/eh_4/laser/delay_line_motor_29"][:])
delay = d_conv(delay, t0=221)

is_data = (is_xray == 1) * (photon_energy > 9651) * (photon_energy < 9700) * (iol < 0.5) * (iou < 0.5) * (iol > 0.) * (iou > 0.) * (nd > -1)

itot = iol[is_data] + iou[is_data]
spd = spd[is_data][itot > 0]
mono = mono[is_data][(itot > 0)]
delay = delay[is_data][(itot > 0)]
is_laser = is_laser[is_data][(itot > 0)]
nd = nd[is_data][(itot > 0)]
itot = itot[itot > 0]
absorp = spd / itot

tags = tags[is_data]
photon_energy = photon_energy[is_data]
iou = iou[is_data]
iol = iol[is_data]


for i in range(0, 2):
    [sum_image_on, spectrum_on], [sum_image_off, spectrum_off] = ut.analysis.get_spectrum_sacla(f["/run_" + run + "/detector_2d_" + str(i + 1) + "/"],
                                                                                                tags, masks=[[is_laser == 1], [is_laser == 0]], thr=thr)

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
