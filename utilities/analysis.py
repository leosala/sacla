import numpy as np
import math
#import cython_utils
from time import time
import sys
import h5py
import pydoc
import pandas as pd


def rebin(a, *args):
    """
    rebin a numpy array
    """
    shape = a.shape
    lenShape = len(shape)
    #factor = np.asarray(shape) / np.asarray(args)
    #print factor
    evList = ['a.reshape('] + ['args[%d],factor[%d],' % (i, i) for i in range(lenShape)] + [')'] + ['.mean(%d)' % (i + 1) for i in range(lenShape)]
    return eval(''.join(evList))


def get_data_daq(filenames, daq_labels, sacla_converter, t0=0, selection=""):
    # create a DataFrame
    df_orig = pd.DataFrame(columns=daq_labels.keys(), )
    failed_filenames = []

    if isinstance(filenames, str):
        filenames = [filenames, ]

    filenames = sorted(filenames)
    for fname in filenames:
        mydict = {}  # temporary dict, where to store data
        
        try:
            f = h5py.File(fname, "r")
            run = int(f.keys()[1].split("_")[1])  # this assumes 1 run per file, as run_XXXXX
            main_dset = f[f.keys()[1]]  
        except:
            print "Error loading file %s: %s" % (fname, sys.exc_info()[1])
            failed_filenames.append(fname)
            continue

        # Loading data from the specified datasets
        for k, v in daq_labels.iteritems():
            if k == "delay":
                # delays are in motor steps
                mydict[k] = sacla_converter.convert("delay", main_dset[v][:], t0=t0)
            elif k == "photon_mono_energy":
                # mono energy settings are in motor steps
                mydict[k] = sacla_converter.convert("energy", main_dset[v][:])
            elif k == "photon_sase_energy":
                mydict[k + "_mean"] = main_dset[v][:].mean()
            else:
                mydict[k] = main_dset[v][:]

        tmp_df = pd.DataFrame(data=mydict)
        tmp_df["run"] = run
        # Append the data to the dataframe
        df_orig = df_orig.append(tmp_df)

    if df_orig.size == 0:
        return df_orig, filenames
        
    # removing failed runs
    for r in failed_filenames:
        filenames.remove(r)

    # round mono energy and delay
    df_orig.photon_mono_energy = np.round(df_orig.photon_mono_energy.values, decimals=4)
    df_orig.delay = np.round(df_orig.delay.values, decimals=1)

    # create total I0 and absorption coefficients
    df_orig["I0"] = df_orig.I0_up + df_orig.I0_down
    df_orig["is_laser"] = (df_orig['laser_status'] == 1)

    # set tag number as index
    df_orig = df_orig.set_index("tags")

    # filtering out garbage
    if selection != "":
        df = df_orig.query(selection)
    else:
        df = df_orig

    # print selection efficiency
    print "\nSelection efficiency"
    sel_eff = pd.DataFrame( {"Total":df_orig.groupby("run").count().ND, 
                             "Selected": df.groupby("run").count().ND, 
                             "Eff.": df.groupby("run").count().ND / df_orig.groupby("run").count().ND})
    print sel_eff

    # checking delay settings
    g = df.groupby(['run', 'delay', 'photon_mono_energy'])
    print "\nEvents per run and delay settings"
    print g.count().I0
    
    return df, filenames
    

def get_energy_from_theta(thetaPosition):
    # Todo: Most probably these variables need to be read out from the control system ...
    theta_coeff = 25000  # in [pulse/mm]
    lSinbar = 275.0  # in [mm]
    theta_offset = -15.0053431  # in [degree]
    dd = 6.270832

    theta = math.asin((thetaPosition / theta_coeff) / lSinbar + math.sin(theta_offset * math.pi / 180.0)) * 180.0 / math.pi - theta_offset
    energy = 12.3984 / ((dd) * math.sin(theta * math.pi / 180.0))

    return energy
