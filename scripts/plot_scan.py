"""
Simple analysis for XAS data. It applies a filter to data, and plot absorption in 
function of a scan parameter (energy, delay, etc). It also plots some control
quantities, e.g. photon energy and I0
"""

# TODO
# remove hardcoded data dir

import matplotlib.pyplot as plt
import pandas as pd
import h5py
import numpy as np
from os import environ
import os
from sys import path, exc_info
# loading some utils
path.append(environ["PWD"] + "/../")
# contains some useful conversions, which can vary from beamtime to beamtime
from utilities import beamtime_converter_201411XX as btc


# Define SACLA quantities - they can change from beamtime to beamtime
IOlow = "/event_info/bl_3/eh_4/photodiode/photodiode_I0_lower_user_7_in_volt"
IOup = "/event_info/bl_3/eh_4/photodiode/photodiode_I0_upper_user_8_in_volt"
PDSample = "/event_info/bl_3/eh_4/photodiode/photodiode_sample_PD_user_9_in_volt"
Mono = "/event_info/bl_3/tc/mono_1_position_theta"
Delay = "/event_info/bl_3/eh_4/laser/delay_line_motor_29"
ND = "/event_info/bl_3/eh_4/laser/nd_filter_motor_26"

# directory containing HDF5 files
# dir = "/work/timbvd/hdf5/"
#dir = "/swissfel/photonics/data/2014-11-26_SACLA_ZnO/hdf5/"

# units
units = {}
units["delay"] = "ps"
units["energy"] = "keV"

# constant quantity during a scan
const_quant = {}
const_quant["delay"] = "energy"
const_quant["energy"] = "delay"


def compute_xas(scan_type, start_run, end_run, data_dir, t0=0):
    """
    load HDF5 files corresponding to runs, filter data and return datasets (using pandas DataFrames)

    :param scan_type: can be any quantity which is configured, e.g. energy, delay
    :param start run: first run of the scan
    :param end_run: last run of the scan
    """

    df = None
    df_conds = None

    index_name = scan_type

    for i in range(int(start_run), int(end_run) + 1):
        # fname = dir + str(i) + "_nompccd.h5"
        fname = data_dir + str(i) + "_roi.h5"
        run = fname.split("/")[-1].replace("_roi", "").replace(".h5", "").replace("_nompccd", "")

        try:
            f = h5py.File(fname, "r")
            tags = f["/run_" + run + "/event_info/tag_number_list"][:]
        except IOError:
            print exc_info()
            continue
        except:
            print exc_info()
            #print "Last good run was %d" % int(i - 1)
            #end_run = str(i - 1)
            #continue
            print "[ERROR] dunno what to do, call support!"
        #break 

        # create dataframes from hdf5 files
        photon_energy = f["/run_" + run + "/event_info/bl_3/oh_2/photon_energy_in_eV"][:]
        is_xray = f["/run_" + run + "/event_info/bl_3/eh_1/xfel_pulse_selector_status"][:]
        is_laser = f["/run_" + run + "/event_info/bl_3/lh_1/laser_pulse_selector_status"][:]
        iol = np.array(f["/run_" + run + IOlow][:])
        iou = np.array(f["/run_" + run + IOup][:])
        spd = np.array(f["/run_" + run + PDSample][:])
        mono = btc.convert("energy", np.array(f["/run_" + run + Mono][:]))
        nd = np.array(f["/run_" + run + ND])
        delay = np.array(f["/run_" + run + "/event_info/bl_3/eh_4/laser/delay_line_motor_29"][:])
        delay = btc.convert("delay", delay, t0=t0)

        # Data filtering - to be changed depending on exp. conditions
        is_data = (is_xray == 1) * (photon_energy > 9600) * (iol < 0.5) * (iou < 0.5) * (iol > 0.01) * (iou > 0.01) * (nd > -1)

        # Applying the filter
        itot = iol[is_data] + iou[is_data]
        spd = spd[is_data][itot > 0]
        mono = mono[is_data][(itot > 0)]
        delay = delay[is_data][(itot > 0)]
        is_laser = is_laser[is_data][(itot > 0)]
        nd = nd[is_data][(itot > 0)]
        itot = itot[itot > 0]
        tags = tags[is_data]
        photon_energy = photon_energy[is_data]
        iou = iou[is_data]
        iol = iol[is_data]
        # Calculating the absorption coeff.
        absorp = spd / itot

        # Create a simple dictionary with the interesting data
        data_df = {"energy": mono, "laser": is_laser, "absorp": absorp, "delay": delay, "ND": nd}

        # Create dataframes
        if df is None:
            df = pd.DataFrame(data_df, )
            df = df.set_index(index_name)
        else:
            df = pd.concat([df, pd.DataFrame(data_df, ).set_index(index_name)])
        # Monitoring experimental conditions - in function of puls number
        if df_conds is None:
            df_conds = pd.DataFrame({"tags": tags, "photon_energy": photon_energy, "I0up": iou, "I0down": iol, }, )
            df_conds = df_conds.set_index("tags")
        else:
            df_conds = pd.concat([df_conds, pd.DataFrame({"tags": tags, "photon_energy": photon_energy, "I0up": iou, "I0down": iol, }, ).set_index("tags")])

    return df, df_conds, end_run


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--outputdir", help="output directory", action="store", default=".")
    parser.add_argument("-d", "--datadir", help="data directory directory", action="store", default= "/swissfel/photonics/data/2014-11-26_SACLA_ZnO/hdf5/")
    parser.add_argument("-l", "--label", help="prefix label for image, ASCII output. Default: test", action="store", default=None)

    parser.add_argument("-n", "--noplot", help="do not plot", action="store_true", default=False)
    parser.add_argument("-s", "--start_run", help="start_run", action="store", default="1")
    parser.add_argument("-e", "--end_run", help="end_run", action="store", default="9999999")
    parser.add_argument("-t", "--scan_type", help="scan type, e.g. energy, delay, ND", action="store", default="energy")
    parser.add_argument("-c", "--check", help="plot check quantities (photon energy etc)", action="store_true", default=False)
    parser.add_argument("-a", "--asciifile", help="Dump plot points in ASCII (tab-separated) files, one per dataset. The files will be called e.g. <label>_laser_on.txt, etc. Default: do not dump", action="store_true", default="")

    args = parser.parse_args()

    # t0
    t0 = 221
    df, df_conds, end_run = compute_xas(args.scan_type, args.start_run, args.end_run, args.datadir, t0=t0)

    if args.check:
        df_conds.plot(subplots=True, linestyle="", marker=".", figsize=(10, 10))

    if args.label is None:
        args.label = "scan-%s_%s" %(args.start_run, end_run)
    #print df
    df_laser_on = df[df["laser"] == 1]
    df_laser_off = df[df["laser"] == 0]

    index_name = args.scan_type
    df_counts_on = df_laser_on.groupby(level=index_name).count()["ND"]
    df_counts_off = df_laser_off.groupby(level=index_name)

    # get averages
    df_on = df_laser_on.mean(level=0)
    df_off = df_laser_off.mean(level=0)
    df_diff = (df_on - df_off)

    # get std_dev
    df_on_std = df_laser_on.std(level=0)
    df_off_std = df_laser_off.std(level=0)
    df_diff_std = np.sqrt((df_on_std) * (df_on_std) + (df_off_std) * (df_off_std))

    # std error
    df_on["absorp_stderr"] = df_laser_on["absorp"].sem(level=index_name)
    df_off["absorp_stderr"] = df_laser_off["absorp"].sem(level=index_name)
    df_diff["absorp_stderr"] = np.sqrt((df_on["absorp_stderr"]) ** 2 + (df_off["absorp_stderr"]) ** 2)

    # plot
    fig = plt.figure(figsize=(10, 7))
    if const_quant.has_key(index_name):
        fig.suptitle("Runs %s - %s, %s = %.3f %s" % (args.start_run, end_run, const_quant[index_name], df[const_quant[index_name]].iloc[0], units[const_quant[index_name]]), fontsize=20)
    else:
        fig.suptitle("Runs %s - %s" % (args.start_run, end_run), fontsize=20)
    a2 = plt.subplot(111)
    df_on["absorp"].plot(label="laser on", color="b", linestyle="-", marker="o", yerr=df_on["absorp_stderr"])
    df_off["absorp"].plot(label="laser off", color="r", linestyle="-", marker="o", yerr=df_off["absorp_stderr"])
    df_diff["absorp"].plot(label="on - off", color="k", linestyle="-", marker="o", yerr=df_diff["absorp_stderr"])
    
    ### This is if you want error bands
    #a2.fill_between(df_on.index.tolist(), (df_on - df_on_std)["absorp"].values.tolist(), (df_on + df_on_std)["absorp"].values.tolist(), alpha=0.8, edgecolor='#4d4dff', facecolor='#4d4dff')
    #a2.fill_between(df_off.index.tolist(), (df_off - df_off_std)["absorp"].values.tolist(), (df_off + df_off_std)["absorp"].values.tolist(), alpha=0.8, edgecolor='#FF6666', facecolor='#FF6666')
    #a2.fill_between(df_diff.index.tolist(), (df_diff - df_diff_std)["absorp"].values.tolist(), (df_diff + df_diff_std)["absorp"].values.tolist(), alpha=0.8, edgecolor='#999999', facecolor='#999999')
    a2.set_xlabel("%s (%s)" %(index_name, units[index_name]))
    a2.set_ylabel("a.u. (with standard errors)")
    plt.legend(title="", loc="best")

    if args.asciifile:
        # add std dev to dataframe
        df_on["absorp_std"] = df_on_std["absorp"]
        df_off["absorp_std"] = df_off_std["absorp"]
        df_diff["absorp_std"] = df_diff_std["absorp"]
        
        # sorting
        df_on = df_on.sort(axis=1)
        df_off = df_off.sort(axis=1)
        df_diff = df_diff.sort(axis=1)
        
        df_on.to_csv(os.path.join(args.outputdir, args.label + "_laser_on.txt"), index=True, sep='\t', header=True, )
        df_off.to_csv(os.path.join(args.outputdir, args.label + "_laser_off.txt"), index=True, sep='\t', header=True, )
        df_diff.to_csv(os.path.join(args.outputdir, args.label + "_on_minus_off.txt"), index=True, sep='\t', header=True, )


    # an example of how to plot reference data contained in ASCII file
    """
    fig2 = plt.figure(figsize=(20, 15))
    a3 = plt.subplot(111)
    df_aps = pd.read_table("/work/timbvd/XANES_APS_ZnO.txt", sep="\t")
    df_aps = df_aps.set_index("allDiff_E_calib")
    df_diff_norm = df_diff["absorp"] / (10 * .13)
    df_diff_std_norm = df_diff_std["absorp"] / (10 * .13)
    df_diff_norm.plot(label="SACLA", linewidth=2., marker="o")
    a3.fill_between(df_diff_norm.index.tolist(), (df_diff_norm - df_diff_std_norm).values.tolist(), (df_diff_norm + df_diff_std_norm).values.tolist(), alpha=0.7, edgecolor='#00ffff', facecolor='#00ffff')
    df_aps["allDiff_N"].plot(label="APS", linewidth=2, linestyle="--", color="k")
    a3.fill_between(df_aps.index.tolist(), (df_aps["allDiff_N"] - df_aps["allDiff_N_err"]).values.tolist(), (df_aps["allDiff_N"] + df_aps["allDiff_N_err"]).values.tolist(), alpha=0.7, edgecolor='#999999', facecolor='#999999')

    plt.legend(title="delay = %d ps" % 2, loc="best")
    """

    plt.savefig(os.path.join(args.outputdir, args.label + ".png"))

    print ""
    if args.asciifile:
        print "ASCII dumps and scan plot saved in: %s" % args.outputdir
        print "See them with e.g.: ls -lah %s*" % os.path.join(args.outputdir, args.label)
    else:
        print "Scan plot saved in: %s" % args.outputdir
        print "See it with e.g.: ls -lah %s*" % os.path.join(args.outputdir, args.label)
    if not args.noplot:
        plt.show()
