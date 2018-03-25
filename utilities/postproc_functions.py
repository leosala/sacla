# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:52:18 2015

@author: VEsp
"""

import pickle as pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import ndimage

from photon_tools.images_processor import ImagesProcessor

plt.style.use('ggplot')

def plotInDataFrame(rname, xdata='delay', ydata='intensity', options=[], selection=''):
    """
    plot data from saved .p dataframe

    INPUT:
        - rname: name of the file
        - xdata, ydata: x and y column in the dataframe
        - options:
            - 'bkg': background subtraction
            - 'I0': I0 normalization
            - 'countErr': errorbars as count error (sqrt(N))
        - selection: filter data: example: 'laser_status == 1'
    """

    DIR = "./analyzed_runs/"
    fname = DIR + rname + ".p"
    df_orig = pickle.load(open(fname, "rb"))

    if selection != "":
        df = df_orig.query(selection)
    else:
        df = df_orig

    x = df[xdata]
    y = df[ydata]

    if 'bkg' in options:
        y = np.subtract(y, df.bkg)

    if 'I0' in options:
        y = np.divide(y, df.I0)

    if 'countErr' in options:
        yCountErr = np.sqrt(y)
        plt.figure(figsize=(8, 8))
        plt.errorbar(x, y, yerr=yCountErr, fmt='o')
    else:
        plt.figure(figsize=(8, 8))
        plt.plot(x, y, 'o')

    plt.xlabel(xdata)
    plt.ylabel(ydata)

    return x, y


def getTTfromCSV(df_orig, CSVname, method, tt_stage_offset=0):
    """
    load timing tool data from its CSV file
    Checks for correspondence between tagnumbers with the file created with
    get_data_daq.
    In case no perfect correspondence is found, only common tagnumbers are kept
    timing tool data are written as a new column in the dataframe df:
    df.tt
    """

    dfCSV = pd.read_csv(CSVname, header=1, index_col=0, names=["derivative", "fit", "dl"])

    # check taglist
    if np.array_equal(df_orig.index, dfCSV.index):
        df = df_orig
        print("TT-data matching: tag list matches")
    else:
        commonTags = set(df_orig.index).intersection(dfCSV.index)
        df = df_orig.loc[commonTags]
        dfCSV = dfCSV.loc[commonTags]
        ratio = float(len(df.index)) / len(df_orig.index)
        string = "\n TT-data matching: " + str(ratio) + " of total tags are matching."
        print(string)

    df["dl_corr"] = df.delay - df.delay_tt_stage - tt_stage_offset

    if "fit" in method:
        df['tt'] = dfCSV.fit
    elif "derivative" in method:
        df['tt'] = dfCSV.derivative
    else:
        print("Please give a correct method: ""fit"" or ""derivative"". ")

#    df_loff = df_orig[df.laser_status == 0]
#    df_loff['tt'] = 0
#    df = df.append(df_loff)

    return df


def bin_tt(df, bin_edges, calibration=-2.8):
    """
    bin data according to the timing tool
    Returns a binned dataframe containing the intensity, the background, I0 and the number
    of shots in each bin
    remark: timing tool data do not include laser off shots
    rem: the calibration is given in fs/pixel
    """

    # create corrected delay
    df['dl_corr'] = df.dl_corr*6.6667 + calibration*df.tt
    bin_size = bin_edges[1]-bin_edges[0]

    df_xon = df[df.x_status == 1]
    df_lon = df_xon[df.laser_status == 1]
    df_loff = df_xon[df.laser_status == 0]

    bin_center = bin_edges[:-1]+0.5*bin_size
    df_out = pd.DataFrame(bin_center, columns=['time'])

    if len(df_lon) != 0:
        binned_int_lon = stats.binned_statistic(df_lon.dl_corr, df_lon.intensity, bins=bin_edges, statistic='mean')
        binned_bkg_lon = stats.binned_statistic(df_lon.dl_corr, df_lon.bkg, bins=bin_edges, statistic='mean')
        binned_I0_lon = stats.binned_statistic(df_lon.dl_corr, df_lon.I0, bins=bin_edges, statistic='mean')
        df_out['intensity_lon'] = binned_int_lon.statistic
        df_out['bkg_lon'] = binned_bkg_lon.statistic
        df_out['I0_lon'] = binned_I0_lon.statistic
    else:
        print('No laser ON shots')

    if len(df_loff) != 0:
        binned_int_loff = stats.binned_statistic(df_loff.dl_corr, df_loff.intensity, bins=bin_edges, statistic='mean')
        binned_bkg_loff = stats.binned_statistic(df_loff.dl_corr, df_loff.bkg, bins=bin_edges, statistic='mean')
        binned_I0_loff = stats.binned_statistic(df_loff.dl_corr, df_loff.I0, bins=bin_edges, statistic='mean')
        df_out['I0_loff'] = binned_I0_loff.statistic
        df_out['bkg_loff'] = binned_bkg_loff.statistic
        df_out['intensity_loff'] = binned_int_loff.statistic
    else:
        print('No laser OFF shots')

    bins_hist = bin_edges/bin_size
    if bins_hist[0] < 0:
        bins_hist = bins_hist - bins_hist[0] + 0.5

    hist = np.histogram(binned_int_lon.binnumber, bins=bins_hist)
    df_out['number_in_bins'] = hist[0]

    return df_out


def bin_motor(df, motor='delay'):
    """
    bin data according to motor, without any timing tool correction
    Basically averages the intensity, bkg, I0 at each motor position. It is made this way, because
    it is basically a copy from the timing tool binning function.
    """

    # create corrected delay
    df['scan_motor'] = df[motor]
    bin_center = df[motor].unique()
    bin_center = sorted(bin_center)
    bin_size = min(np.diff(bin_center))

    bin_edges = np.append(bin_center[0]-0.5*bin_size, bin_center+0.5*bin_size)

    df_xon = df[df.x_status == 1]
    df_lon = df_xon[df.laser_status == 1]
    df_loff = df_xon[df.laser_status == 0]

    df_out = pd.DataFrame(bin_center, columns=[motor])

    if len(df_lon) != 0:
        binned_int_lon = stats.binned_statistic(df_lon.scan_motor, df_lon.intensity, bins=bin_edges, statistic='mean')
        binned_bkg_lon = stats.binned_statistic(df_lon.scan_motor, df_lon.bkg, bins=bin_edges, statistic='mean')
        binned_I0_lon = stats.binned_statistic(df_lon.scan_motor, df_lon.I0, bins=bin_edges, statistic='mean')
        df_out['intensity_lon'] = binned_int_lon.statistic
        df_out['bkg_lon'] = binned_bkg_lon.statistic
        df_out['I0_lon'] = binned_I0_lon.statistic

        binned_int_lon_std = stats.binned_statistic(df_lon.scan_motor, df_lon.intensity, bins=bin_edges, statistic='std')
        binned_bkg_lon_std = stats.binned_statistic(df_lon.scan_motor, df_lon.bkg, bins=bin_edges, statistic='std')
        binned_I0_lon_std = stats.binned_statistic(df_lon.scan_motor, df_lon.I0, bins=bin_edges, statistic='std')
        df_out['intensity_lon_std'] = binned_int_lon_std.statistic
        df_out['bkg_lon_std'] = binned_bkg_lon_std.statistic
        df_out['I0_lon_std'] = binned_I0_lon_std.statistic
    else:
        print('No laser ON shots')

    if len(df_loff) != 0:
        binned_int_loff = stats.binned_statistic(df_loff.scan_motor, df_loff.intensity, bins=bin_edges, statistic='mean')
        binned_bkg_loff = stats.binned_statistic(df_loff.scan_motor, df_loff.bkg, bins=bin_edges, statistic='mean')
        binned_I0_loff = stats.binned_statistic(df_loff.scan_motor, df_loff.I0, bins=bin_edges, statistic='mean')
        df_out['I0_loff'] = binned_I0_loff.statistic
        df_out['bkg_loff'] = binned_bkg_loff.statistic
        df_out['intensity_loff'] = binned_int_loff.statistic
        binned_int_loff_std = stats.binned_statistic(df_loff.scan_motor, df_loff.intensity, bins=bin_edges, statistic='std')
        binned_bkg_loff_std = stats.binned_statistic(df_loff.scan_motor, df_loff.bkg, bins=bin_edges, statistic='std')
        binned_I0_loff_std = stats.binned_statistic(df_loff.scan_motor, df_loff.I0, bins=bin_edges, statistic='std')
        df_out['I0_loff_std'] = binned_I0_loff_std.statistic
        df_out['bkg_loff_std'] = binned_bkg_loff_std.statistic
        df_out['intensity_loff_std'] = binned_int_loff_std.statistic
    else:
        print('No laser OFF shots')

    return df_out


def bin_tt_COM(df, bin_edges, rname, fname, calibration=0.01, roi=[[235, 270], [500, 540]]):
    """
    Bin data according to the timing tool and perform a center of mass analysis of the roi
    This script is somewhat redundant with the image analysis, as it loops again through all the images.
    """

    # create corrected delay
    df['dl_corr'] = df.delay + calibration*df.tt
    bin_size = bin_edges[1]-bin_edges[0]

    df_xon = df[df.x_status == 1]
    df_lon = df_xon[df.laser_status == 1]
    df_loff = df_xon[df.laser_status == 0]

    bin_center = bin_edges[:-1]+0.5*bin_size
    df_out = pd.DataFrame(bin_center, columns=['time'])

    if len(df_lon) != 0:
        binned_int_lon = stats.binned_statistic(df_lon.dl_corr, df_lon.intensity, bins=bin_edges, statistic='mean')
        binned_bkg_lon = stats.binned_statistic(df_lon.dl_corr, df_lon.bkg, bins=bin_edges, statistic='mean')
        binned_I0_lon = stats.binned_statistic(df_lon.dl_corr, df_lon.I0, bins=bin_edges, statistic='mean')
        df_out['intensity_lon'] = binned_int_lon.statistic
        df_out['bkg_lon'] = binned_bkg_lon.statistic
        df_out['I0_lon'] = binned_I0_lon.statistic
    else:
        print('No laser ON shots')

    if len(df_loff) != 0:
        binned_int_loff = stats.binned_statistic(df_loff.dl_corr, df_loff.intensity, bins=bin_edges, statistic='mean')
        binned_bkg_loff = stats.binned_statistic(df_loff.dl_corr, df_loff.bkg, bins=bin_edges, statistic='mean')
        binned_I0_loff = stats.binned_statistic(df_loff.dl_corr, df_loff.I0, bins=bin_edges, statistic='mean')
        df_out['I0_loff'] = binned_I0_loff.statistic
        df_out['bkg_loff'] = binned_bkg_loff.statistic
        df_out['intensity_loff'] = binned_int_loff.statistic
    else:
        print('No laser OFF shots')

    """
    COM analysis
        COM analysis loops through the bins and load the images corresponding for each bin.
        The COM of the averaged images in the bin is taken and written in the df_out dataframe
    """
    binnumber = binned_int_lon.binnumber
    peakCOM = np.zeros([len(df_out.time), 2])

    dataset_name = "/run_" + rname + "/detector_2d_1"
    ip = ImagesProcessor(facility="SACLA")

    ip.flatten_results = True
    ip.set_dataset(dataset_name)

    ip.add_preprocess("set_roi", args={'roi': roi})
    ip.add_analysis("get_mean_std")

    for ii in range(len(df_out.time)):
        n = ii+1
        ismember = (binnumber == n)

        tagList = df.index[ismember]
        results = ip.analyze_images(fname, n=-1, tags=tagList)

        if 'images_mean' in results:
            peakCOM[ii, :] = ndimage.measurements.center_of_mass(results['images_mean'])
        else:
            peakCOM[ii, :] = np.NaN

        del results
        print(('bin number %s' % n))

    df_out['COMx'] = peakCOM[:, 0]
    df_out['COMy'] = peakCOM[:, 1]

    return df_out

