import matplotlib.pyplot as plt
import pandas as pd
import h5py
import numpy as np
import math

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

    theta = math.asin((theta_position / theta_coeff) / lSinbar + math.sin(theta_offset * math.pi / 180.0)) * 180.0 / math.pi - theta_offset
    energy = (12.3984 / (dd)) / math.sin(theta * math.pi / 180.0)

    return energy  # , units


def get_delay_from_pulse(pulse, t0=0):
    """"""
    magic_factor = 6.66713134 / 1000.
    return ((float(pulse) * magic_factor) - t0)

dir = "/work/timbvd/hdf5/"
#t0 = 221  # delay t0


def compute_xas(scan_type, start_run, end_run, t0=0):
    df = None
    df_conds = None

    index_name = scan_type

    for i in range(int(start_run), int(end_run) + 1):
        fname = dir + str(i) + "_nompccd.h5"
        run = fname.split("/")[-1].replace("_roi", "").replace(".h5", "").replace("_nompccd", "")

        try:
            f = h5py.File(fname, "r")
            tags = f["/run_" + run + "/event_info/tag_number_list"][:]
        except:
            print "Last good run was %d" % int(i - 1)
            end_run = str(i - 1)
            break

        photon_energy = f["/run_" + run + "/event_info/bl_3/oh_2/photon_energy_in_eV"][:]
        is_xray = f["/run_" + run + "/event_info/bl_3/eh_1/xfel_pulse_selector_status"][:]
        is_laser = f["/run_" + run + "/event_info/bl_3/lh_1/laser_pulse_selector_status"][:]
        t_conv = np.vectorize(get_energy_from_theta)
        d_conv = np.vectorize(get_delay_from_pulse)

        iol = np.array(f["/run_" + run + IOlow][:])
        iou = np.array(f["/run_" + run + IOup][:])
        spd = np.array(f["/run_" + run + PDSample][:])
        mono = t_conv(np.array(f["/run_" + run + Mono][:]))
        nd = np.array(f["/run_" + run + ND])
        delay = np.array(f["/run_" + run + "/event_info/bl_3/eh_4/laser/delay_line_motor_29"][:])
        delay = d_conv(delay, t0=t0)

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

        data_df = {"energy": mono, "laser": is_laser, "absorp": absorp, "delay": delay, "ND": nd}

        if df is None:
            df = pd.DataFrame(data_df, )
            df = df.set_index(index_name)
        else:
            df = pd.concat([df, pd.DataFrame(data_df, ).set_index(index_name)])
        if df_conds is None:
            df_conds = pd.DataFrame({"tags": tags,
                                     "photon_energy": photon_energy,
                                     "I0up": iou,
                                     "I0down": iol,
                                 }, )
            df_conds = df_conds.set_index("tags")
        else:
            df_conds = pd.concat([df_conds, pd.DataFrame({"tags": tags, "photon_energy": photon_energy, "I0up": iou, "I0down": iol, }, ).set_index("tags")])
        
    return df, df_conds



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    #parser.add_argument("filename", help="Name of the .csv file with the average scan info", nargs='+')
    parser.add_argument("-o", "--outputdir", help="output directory", action="store", default=".")
    parser.add_argument("-p", "--plot", help="plot, do not save", action="store_true")
    parser.add_argument("-s", "--start_run", help="start_run", action="store", default="1")
    parser.add_argument("-e", "--end_run", help="end_run", action="store", default="9999999")
    parser.add_argument("-t", "--scan_type", help="scan type, e.g. energy, delay, ND", action="store", default="energy")
    parser.add_argument("-c", "--check", help="plot check quantities (photon energy etc)", action="store_true", default=False)

    args = parser.parse_args()
    df, df_conds = compute_xas(args.scan_type, args.start_run, args.end_run)

    if args.check:
        df_conds.plot(subplots=True, linestyle="", marker=".", figsize=(10, 10))

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle("Runs %s - %s" % (args.start_run, args.end_run))
    a2 = plt.subplot(111)
    df_on = df[df["laser"] == 1].mean(level=0)
    df_off = df[df["laser"] == 0].mean(level=0)
    df_diff = 10 * (df_on - df_off)
    df_on_std = df[df["laser"] == 1].std(level=0)
    df_off_std = df[df["laser"] == 0].std(level=0)

    df_diff_std = np.sqrt((df_on_std) * (df_on_std) + (df_off_std) * (df_off_std))
 
    df_on["absorp"].plot(label="laser on", color="b", linestyle="-", marker="o")
    df_off["absorp"].plot(label="laser off", color="r", linestyle="-", marker="o")
    df_diff["absorp"].plot(label="10x on - off", color="k", linestyle="-", marker="o")
    a2.fill_between(df_on.index.tolist(), (df_on - df_on_std)["absorp"].values.tolist(), (df_on + df_on_std)["absorp"].values.tolist(), alpha=0.5, edgecolor='#00ffff', facecolor='#00ffff')
    a2.fill_between(df_off.index.tolist(), (df_off - df_off_std)["absorp"].values.tolist(), (df_off + df_off_std)["absorp"].values.tolist(), alpha=0.5, edgecolor='#f5deb3', facecolor='#f5deb3')

    a2.fill_between(df_diff.index.tolist(), (df_diff - df_diff_std)["absorp"].values.tolist(), (df_diff + df_diff_std)["absorp"].values.tolist(), alpha=0.5, edgecolor='#eee9e9', facecolor='#eee9e9')


    plt.legend(title="t0 = %d ps" % t0, loc="best")
    plt.show()
