# -*- coding: utf-8 -*-
"""
Created on Wed May 18 23:25:48 2016

@author: VEsp
"""

import cPickle as pickle

import matplotlib.pyplot as plt
import numpy as np

import functions_SACLA_may2016 as fsacla

plt.style.use('ggplot')


def loadData(rname, useTT=0): 
    # Loading SACLA tools 
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

    # data directories and names
    DIR = "/home/usov_i/SACLA Dec2015/python_scripts2016/data/"
    CSVDIR = "/home/usov_i/SACLA Dec2015/python_scripts2016/data/"
    fname = DIR + rname + ".h5"

    df, fnames = sacla_utils.analysis.get_data_daq(fname, daq_labels, sacla_utils.beamtime_converter_201605XX)

    saveDir = "/home/usov_i/SACLA Dec2015/python_scripts2016/analyzed_runs/"
    imgAnaName = saveDir + "imgAna/" + rname + "_v2" + ".p"
    imgAna = pickle.load(open(imgAnaName, "rb"))
    
    df["intensity"] = imgAna["intensity"]
    df["bkg"] = imgAna["bkg"]
    
    # filter on I0. Good values given by beamline scientist in SACLA: 0.005 < I0 < 0.9
    df_orig = df
    df = df[df.I0 < 0.9]
    df = df[df.I0 > 0.005]
    ratio = float(len(df))/len(df_orig)
    print("\n I0 filter: " + str(ratio))

    tt_stage_offset = 30000  # just to make number more readable
    if useTT == 1:
        df = fsacla.getTTfromCSV(df, CSVDIR + rname + ".csv", "derivative", tt_stage_offset)
    
    df.to_pickle(saveDir + rname + "_may2016" + ".p")
    
    return df
