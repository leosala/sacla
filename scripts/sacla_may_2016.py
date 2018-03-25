# -*- coding: utf-8 -*-
"""
This notebook presents a very quick overview of the data conversion and analysis that was developed and used during
the beamtime in SACLA in May 2016.
Don’t hesitate to contact me if you have questions (vincent.esposito@psi.ch)
"""

import os
import sys

dir_path = os.getcwd() + '/../../'

sys.path.append(dir_path + 'sacla')
sys.path.append(dir_path + 'photon_tools')

import matplotlib.pyplot as plt
import numpy as np
import pickle as pickle
from utilities import postproc_functions as fsacla
from photon_tools.images_processor import ImagesProcessor
import utilities as sacla_utils

# Define SACLA quantities - they can change from beamtime to beamtime
daq_labels = {}
daq_labels["I0_down"] = "event_info/PD_I0/lower_in_volt"
daq_labels["I0_up"] = "event_info/PD_I0/upper_in_volt"
# daq_labels["I0_left"] = "event_info/PD_I0/left_in_volt"
# daq_labels["I0_right"] = "event_info/PD_I0/right_in_volt"
daq_labels["I0_down_gainC"] = "event_info/PD_I0/lower_gainC"
daq_labels["I0_up_gainC"] = "event_info/PD_I0/upper_gainC"
# daq_labels["I0_left"] = "event_info/PD_I0/left_in_volt"
# daq_labels["I0_right"] = "event_info/PD_I0/right_in_volt"
# daq_labels["TFY"] = "event_info/bl_3/eh_4/photodiode/photodiode_sample_PD_user_9_in_volt"
# daq_labels["chan_cut_mono_crysta1_theta"] = "event_info/monochrom/crystal1_theta"
# daq_labels["chan_cut_mono_crysta2_theta"] = "event_info/monochrom/crystal2_theta"
daq_labels["delay"] = "event_info/OPT/delay"
daq_labels["delay_tt_stage"] = "event_info/TimingTool/delay"
daq_labels["ND"] = "event_info/OPT/ND"
daq_labels["2theta"] = "event_info/diffractometer/2theta"
daq_labels["theta"] = "event_info/diffractometer/theta"
daq_labels["chi"] = "event_info/diffractometer/chi"
daq_labels["phi"] = "event_info/diffractometer/phi"
daq_labels["mirror_tilt"] = "event_info/OPT/mirror_tilt"
daq_labels["mirror_rotation"] = "event_info/OPT/mirror_rotation"
# daq_labels["photon_sase_energy"] = "event_info/bl_3/oh_2/photon_energy_in_eV"
daq_labels["x_status"] = "event_info/bl_3/eh_1/xfel_pulse_selector_status"
# daq_labels["x_shut"] = "event_info/bl_3/shutter_1_open_valid_status"
daq_labels["laser_status"] = "event_info/bl_3/lh_1/laser_pulse_selector_status"
daq_labels["tags"] = "event_info/tag_number_list"
daq_labels["gonio_z"] = "event_info/goniometer/g_z"

# Define test data runs to be processed
runs = [439064, 439065]  # delay
# runs = [439020]  # mirror_tilt
# runs = [439015]  # phi

# Choose whether imgAna, loadData and timing tool analysis has to be performed
motor = 'delay'

imgAna = 1
loadData = 1
useTT = 1
plotting = 1

fig_num = 132

n = -1

# data directories and names
DIR = "<base_path>/data/"
CSVDIR = "<base_path>/data/"
saveDir = "<base_path>/analyzed_runs/"
img_save_dir = "<base_path>/analyzed_runs/imgAna/"

# Continue execution whenever a figure is created (they will be shown by the end of the script run)
plt.ion()

# INPUT PARAMETERS
thr = 50  # pixel's threshold value
roi = [[450, 520], [240, 280]]  # SL [[xmin xmax], [ymin ymax]]
#  roi = [[420, 470], [190, 230]]  # Bragg Peak [[xmin xmax], [ymin ymax]]
#  bkgRoi = np.array(roi) #+ np.array([[-40, 40], [-40, 40]])
bkgRoi = np.array(roi)

# create ImagesProcessor object
ip = ImagesProcessor(facility="SACLA")

# if you want a flat dict as a result
ip.flatten_results = True

# PREPROCESS FUNCTIONS (bkg sub, masks, ...)
# (comment out for loading a background image)
dark = np.load('/home/usov_i/SACLA Dec2015/python_scripts2016/analysis/dark_439011and02comb.npy')
ip.add_preprocess("subtract_correction", args={"sub_image": dark})
ip.add_preprocess("set_thr", args={"thr_low": thr})

# ANALYSIS FUNCTIONS
ip.add_analysis("get_mean_std")  # , args={'thr_low': thr})
bins = np.arange(-50, 600, 2)
ip.add_analysis("get_histo_counts", args={'bins': bins, 'roi': roi})
ip.add_analysis("roi_bkgRoi", args={'roi': roi, 'bkg_roi': bkgRoi})

for run in runs:
    rname = str(run)
    fname = DIR + rname + ".h5"
    print(('\nAnalyzing run ' + rname + '\n'))
    """
    Analyze images and integrate roi and bkgRoi. Can take a lot of time
    The results are saved in a pickle file in the folder analyzed_runs/imgAna.
    To add prepocess, analysis functions, open the dataAna.imgAna function.
    """
    if imgAna:
        dataset_name = "/run_" + rname + "/detector_2d_1"

        ip.set_dataset(dataset_name, remove_preprocess=False)

        # run the analysis
        results = ip.analyze_images(fname, n=n)

        # plot results
        imgs = results["images_mean"]
        plt.figure(figsize=(8, 8))
        plt.subplot2grid((2, 2), (0, 0), rowspan=2)
        plt.imshow(imgs)
        #     plt.imshow(imgs[bkgRoi[0][0]:bkgRoi[0][1], bkgRoi[1][0]:bkgRoi[1][1]], aspect=0.5,
        #                extent=(bkgRoi[1][0], bkgRoi[1][1], bkgRoi[0][0], bkgRoi[0][1]), interpolation="none")
        plt.subplot2grid((2, 2), (0, 1))
        plt.imshow(imgs[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]], aspect=0.5,
                   extent=(roi[1][0], roi[1][1], roi[0][1], roi[0][0]), interpolation="none")
        plt.title('ROI')

        # FOR DARKFRAME save to npy:
        # np.save('/home/esposiv/python_scripts2016/analysis/dark_2016_388662_testsave.npy', imgs)

        plt.subplot2grid((2, 2), (1, 1))
        #    plt.figure(figsize=(7, 7))
        plt.bar(bins[:-1], results["histo_counts"], log=True, width=5)
        plt.show()

        # save data as a pickle file
        output = open(img_save_dir + rname + '_v2' + ".p", "wb")
        pickle.dump(results, output)
        output.close()

    """
    Load the data from the hdf5 file and append the roi and bkgRoi intensities calculated above
    The results are saved in a pickle file in the folder analyzed_runs.
    To change daq_labels, filters, etc, open the dataAna.loadData function.
    """
    if loadData:
        df, fnames = sacla_utils.analysis.get_data_daq(fname, daq_labels, sacla_utils.beamtime_converter_201605XX)

        imgAnaName = img_save_dir + rname + "_v2" + ".p"
        imgAna = pickle.load(open(imgAnaName, "rb"))

        df["intensity"] = imgAna["intensity"]
        df["bkg"] = imgAna["bkg"]

        # filter on I0. Good values given by beamline scientist in SACLA: 0.005 < I0 < 0.9
        df_orig = df
        df = df[df.I0 < 0.9]
        df = df[df.I0 > 0.005]
        ratio = float(len(df)) / len(df_orig)
        print(("\n I0 filter: " + str(ratio)))

        tt_stage_offset = 30000  # just to make number more readable
        if useTT == 1:
            df = fsacla.getTTfromCSV(df, CSVDIR + rname + ".csv", "derivative", tt_stage_offset)

        df.to_pickle(saveDir + rname + "_may2016" + ".p")

    """ Appends all dataframes together in a big dataframe df_total """
    file = saveDir + str(run) + "_may2016" + ".p"
    if 'df_total' in locals():
        df_total = df_total.append(pickle.load(open(file, "rb")))
    else:
        df_total = pickle.load(open(file, "rb"))

# Filter intensity values that are out of range
# df_total = df_total[df_total.intensity < 160000]

# Rebinning according to the timing tool corrected values
if useTT:
#     bin_edges = np.linspace(-550000, 1100000, 150) # [fs]
    bin_edges = np.arange(-8000, 20000, 40)  # [fs]
    df_binned = fsacla.bin_tt(df_total[df_total.laser_status == 1], bin_edges, calibration=-2.8)

# Average without timing tool
df_ave = fsacla.bin_motor(df_total, motor=motor)

# Plotting
if plotting:
    plt.figure(fig_num, figsize=(15, 12))
    if useTT:
        time = df_binned.time
    #     on =  (df_binned.intensity_lon-df_binned.bkg_lon)/df_binned.I0_lon
    #     on_raw = df_binned.intensity_lon-df_binned.bkg_lon
        on = df_binned.intensity_lon/df_binned.I0_lon
        on_raw = df_binned.intensity_lon
        plt.subplot(3, 2, 1)
        plt.title('Run ' + str(runs[0]) + ' timing tool', fontsize=12)
        plt.plot((time+700)/1000, on, '-o', label='laser on')
        plt.xlabel('time [ps]')
        plt.subplot(3, 2, 2)
        plt.title('Run ' + str(runs[0]) + ' timing tool, no I0', fontsize=12)
        plt.plot((time+700)/1000, on_raw, '-o', label='laser on')
        plt.xlabel('time [ps]')

    if motor is 'delay':
        motor = df_ave[motor]*6.66667/1000
    else:
        motor = df_ave[motor]

    if 'intensity_lon' in df_ave:
    #     on =  (df_ave.intensity_lon-df_ave.bkg_lon)/df_ave.I0_lon
    #     on_raw =  (df_ave.intensity_lon-df_ave.bkg_lon)
        on = df_ave.intensity_lon/df_ave.I0_lon
        on_raw = df_ave.intensity_lon

        plt.subplot(3, 2, 3)
        plt.title('Run ' + str(runs[0]) + ' no timing tool', fontsize=12)
        plt.plot(motor, on, '-o', label='laser on')
        plt.xlabel('motor pos. [pulses]')
        plt.subplot(3, 2, 4)
        plt.title('Run ' + str(runs[0]) + ' no timing tool, no I0', fontsize=12)
        plt.plot(motor, on_raw, '-o', label='laser on')
        plt.xlabel('motor pos. [pulses]')

    if 'intensity_loff' in df_ave:
    #     off =  (df_ave.intensity_loff-df_ave.bkg_loff)/df_ave.I0_loff
    #     off_raw =  df_ave.intensity_loff-df_ave.bkg_loff
        off = df_ave.intensity_loff/df_ave.I0_loff
        off_raw = df_ave.intensity_loff

        plt.subplot(3, 2, 3)
        plt.plot(motor, off, '-o', label='laser off')
        plt.xlabel('motor pos. [pulses]')
        plt.legend(loc='lower left')
        plt.subplot(3, 2, 4)
        plt.plot(motor, off_raw, '-o', label='laser off')
        plt.legend(loc='lower left')
        plt.xlabel('motor pos. [pulses]')

        if 'intensity_lon' in df_ave:
            plt.subplot(3, 2, 5)
            plt.title('Run ' + str(runs[0]) + ' difference', fontsize=12)
            plt.plot(motor, (on-off), '-o', label='diff')
            plt.xlabel('motor pos. [pulses]')
            plt.legend(loc='lower left')
            plt.subplot(3, 2, 6)
            plt.title('Run ' + str(runs[0]) + ' difference, no I0', fontsize=12)
            plt.plot(motor, on_raw-off_raw, '-o',label='diff')
            plt.legend(loc='lower left')
            plt.xlabel('motor pos. [pulses]')

# Keep the final figure open
plt.ioff()
plt.show()
