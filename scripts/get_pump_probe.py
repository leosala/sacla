import h5py
from time import time
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from math import asin, sin, pi

# loading some utils
sys.path.append(os.environ["PWD"] + "/../")
from utilities import beamtime_converter_201411XX as btc
#from anbox import analysis_loader
import create_spectra
# from utilities.analysis import rebin

IOlow = "/event_info/bl_3/eh_4/photodiode/photodiode_I0_lower_user_7_in_volt"
IOup = "/event_info/bl_3/eh_4/photodiode/photodiode_I0_upper_user_8_in_volt"
PDSample = "/event_info/bl_3/eh_4/photodiode/photodiode_sample_PD_user_9_in_volt"
Mono = "/event_info/bl_3/tc/mono_1_position_theta"
Delay = "/event_info/bl_3/eh_4/laser/delay_line_motor_29"
ND = "/event_info/bl_3/eh_4/laser/nd_filter_motor_26"

if __name__ == "__main__":
    fname = sys.argv[1]
    is_roi = True  # is a ROI'd file?
    thr = 70  # APU counting threshold

    #print fname, fname.find("roi")
    if fname.find("roi") == -1:
        is_roi = False

    run = fname.split("/")[-1].replace("_roi", "").replace(".h5", "")

    f = h5py.File(fname, "r")
    #print f["/run_" + run + "/"].keys()
    #print f.keys()

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

    iol = np.array(f["/run_" + run + IOlow][:])
    iou = np.array(f["/run_" + run + IOup][:])
    spd = np.array(f["/run_" + run + PDSample][:])
    mono = btc.convert("energy", np.array(f["/run_" + run + Mono][:]))
    nd = np.array(f["/run_" + run + ND])
    delay = np.array(f["/run_" + run + "/event_info/bl_3/eh_4/laser/delay_line_motor_29"][:])
    delay = btc.convert("delay", delay)

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
    image_sum = []
    if is_roi:
        for i in range(2):
            image_sum.append(f["/run_" + run + "/detector_2d_" + str(i + 1) + "/image_avg"][:])

    f.close()  # must do for multiprocessing
    #rois = [[[0, 1024], [440, 512]], [[0, 1024], [460, 490]]]  # X, Y
    rois = [[], []]
    plugin_conf = {}
    for i in range(0, 2):
        plugin_conf['create_spectra'] = {"roi": [[0, 1024], [325, 335]]}
        #algo = analysis_loader.load("create_spectra")
        algo = create_spectra.CreateSpectraPumpProbe()
        algo.tags_list = tags
        algo.dst_name = "/run_" + run + "/detector_2d_" + str(i + 1) + "/"
        algo.fname = fname
        algo.masks = is_laser == 1
        algo.thr = thr
        [sum_image_on, spectrum_on] = algo.run()
        algo.masks = is_laser == 0
        [sum_image_off, spectrum_off] = algo.run()

        if is_roi:
            ax.append(plt.subplot(2, 3, 1 + 3 * (i)))
            plt.imshow(image_sum[i])
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
