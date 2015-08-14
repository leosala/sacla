# # Sacla - RIXS example
# 
# In this example, some simple code is provided as an example to produce some Resonant Inelastic X-Ray Scattering (RIXS) maps.
# 
# The basic steps to be performed are:
# - select a suitable set of runs
# - scan for finding the scanned monochromator energies
# - produce the spectra for each energy, and the on/off maps
# 
# *The aim of this tutorial is not to produce extremely efficient code, but code as simple and as fast as possible 
# to support the data quality evaluation during beamtime and after.*
# 


import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import pandas as pd

# Loading SACLA tools 
#SACLA_LIB = "../"
SACLA_LIB = "/swissfel/photonics/software/python-code/sacla/"
sys.path.append(SACLA_LIB)
import utilities as ut

# specific converters for the 2014-11 data taking. These should be customized per each beamtime!
from utilities import beamtime_converter_201411XX as sacla_converter

# directory containing ROI'd hdf5 files
DIR = "/swissfel/photonics/data/2014-11-26_SACLA_ZnO/hdf5/"
dark_file = "/swissfel/photonics/data/2014-11-26_SACLA_ZnO/dark/dark_256635.h5"

# Then, we define:
# * the SACLA datasets
# * $t_0$

# Define SACLA quantities - they can change from beamtime to beamtime
daq_labels = {}
daq_labels["I0_down"] = "event_info/bl_3/eh_4/photodiode/photodiode_I0_lower_user_7_in_volt"
daq_labels["I0_up"] = "event_info/bl_3/eh_4/photodiode/photodiode_I0_upper_user_8_in_volt"
daq_labels["TFY"] = "event_info/bl_3/eh_4/photodiode/photodiode_sample_PD_user_9_in_volt"
daq_labels["photon_mono_energy"] = "event_info/bl_3/tc/mono_1_position_theta"
daq_labels["delay"] = "event_info/bl_3/eh_4/laser/delay_line_motor_29"
daq_labels["ND"] = "event_info/bl_3/eh_4/laser/nd_filter_motor_26"
daq_labels["photon_sase_energy"] = "event_info/bl_3/oh_2/photon_energy_in_eV"
daq_labels["x_status"] = "event_info/bl_3/eh_1/xfel_pulse_selector_status"
daq_labels["x_shut"] = "event_info/bl_3/shutter_1_open_valid_status"
daq_labels["laser_status"] = "event_info/bl_3/lh_1/laser_pulse_selector_status"
daq_labels["tags"] = "event_info/tag_number_list"

# the t0, to be found experimentally
t0 = 220.86


def get_data_df(dataset_name, runs, selection=""):
    # create a DataFrame
    df_orig = pd.DataFrame(columns=daq_labels.keys(), )

    failed_runs = []
    runs = sorted(runs)
    for run in runs:
        mydict = {}  # temporary dict, where to store data
        fname = DIR + str(run) +"_roi.h5"  # the file name
        try:
            f = h5py.File(fname, "r")
            main_dset = f["run_" + str(run)]
        except:
            print "Error loading run %s: %s" % (run, sys.exc_info[1])
            failed_runs.append(run)
            pass

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

    # removing failed runs
    for r in failed_runs:
        runs.remove(r)

    # round mono energy and delay
    df_orig.photon_mono_energy = np.round(df_orig.photon_mono_energy.values, decimals=4)
    df_orig.delay = np.round(df_orig.delay.values, decimals=1)

    # create total I0 and absorption coefficients
    df_orig["I0"] = df_orig.I0_up + df_orig.I0_down
    df_orig["absorp"] = df_orig.TFY / df_orig.I0
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
    print g.count().TFY
    
    return df, runs


def compute_rixs_spectra(dataset_name, df, thr_low=0, thr_hi=999999, ):
    # In principle, a single run can contain *multiple mono settings*, so we need to load data from all the runs, and the group them by mono energy. `Pandas` can help us with that...
    # We load all data from files, place it in a `DataFrame`, and then add some useful derived quantities. At last, we use `tags` as index for the `DataFrame`

    runs = sorted(df.run.unique())
    print runs

    # label for ascii output dump
    out_label = "rixs_" + runs[0] + "-" + runs[-1]

    delay = df.delay.unique()
    if len(delay) > 1:
        print "More than one delay settings in the selected run range, exiting"
        sys.exit(-1)

    print "\nAvailable energy settings"
    print df.photon_mono_energy.unique(), "\n"

    # Now we can run the analysis. For each energy value and each run, a *list of tags* is created, 
    # such that events have the same mono energy and they are part of the same run (as each run is in a separated file). 
    # For each of these lists, we run the `AnalysisProcessor` and create the required spectra, for laser on and off. 

    # the mono energies contained in the files
    energies_list = sorted(df.photon_mono_energy.unique().tolist())
    fnames = [DIR + str(run) +"_roi.h5" for run in runs]

    # The AnalysisProcessor
    an = ut.analysis.AnalysisProcessor()
    # if you want a flat dict as a result
    an.flatten_results = True

    # add analysis
    an.add_analysis("image_get_spectra", args={'axis': 1, 'thr_low': thr_low, 'thr_hi': thr_hi})
    an.add_analysis("image_get_mean_std", args={'thr_low': thr_low})
    bins = np.arange(-150, 1000, 5)
    an.add_analysis("image_get_histo_adu", args={'bins': bins})
    an.set_sacla_dataset(dataset_name)

    # run the analysis
    n_events = -1
    spectrum_on = None
    spectrum_off = None

    # multiprocessing import
    from multiprocessing import Pool
    from multiprocessing.pool import ApplyResult

    # initialization of the RIXS maps. Element 0 is laser_on_ element 1 is laser_off
    rixs_map = [np.zeros((len(energies_list), 1024)), np.zeros((len(energies_list), 1024))]
    rixs_map_std = [np.zeros((len(energies_list), 1024)), np.zeros((len(energies_list), 1024))]

    n_events = -1
    spectrum = [None, None]
    total_results = {}
    events_per_energy = [{}, {}]

    for i, energy in enumerate(energies_list):
        async_results = []  # list for results

        events_per_energy[0][energy] = 0
        events_per_energy[1][energy] = 0
        energy_masks = []
        # creating the pool
        pool = Pool(processes=8)
        # looping on the runs
        for j, run in enumerate(runs):
            df_run = df[df.run == run]
            energy_masks.append(df_run[df_run.photon_mono_energy == energy])
            # apply the analysis 
            async_results.append(pool.apply_async(an, (fnames[j], n_events, energy_masks[j].index.values)))

        # closing the pool
        pool.close()

        # waiting for all results
        results = [r.get() for r in async_results]
        print "Got results for energy", energy

        # producing the laser on/off maps
        for j, run in enumerate(runs):

            if not total_results.has_key(run):
                total_results[run] = {}

            if not results[j].has_key("spectra"):
                continue

            df_run = df[df.run == run]
            energy_mask = energy_masks[j]
            laser_masks = [None, None]
            if n_events != -1:
                laser_masks[0] = energy_mask.is_laser.values[:n_events]
            else:
                laser_masks[0] = energy_mask.is_laser.values
            laser_masks[1] = ~laser_masks[0]

            for laser in [0, 1]:
                norm = np.count_nonzero(~np.isnan(results[j]["spectra"][laser_masks[laser]][:, 0]))
                events_per_energy[laser][energy] += norm
                spectrum = np.nansum((results[j]["spectra"][laser_masks[laser]].T / df_run[laser_masks[laser]].I0.values).T, axis=0)
                spectrum_events = np.nansum(results[j]["spectra"][laser_masks[laser]], axis=0)
                rixs_map[laser][energies_list.index(energy)] += spectrum
                rixs_map_std[laser][energies_list.index(energy)] += spectrum_events
            
            total_results[run][energy] = {}
            total_results[run][energy]["results"] = results[j]
            total_results[run][energy]["laser_on"] = laser_masks[0]

    for laser in [0, 1]:
        for energy in events_per_energy[0].keys():
            rixs_map[laser][energies_list.index(energy)] /= events_per_energy[laser][energy]
    
        rixs_map_std[laser] = rixs_map[laser] / np.sqrt(rixs_map_std[laser])
        np.savetxt("%s_map_%s_%dps.txt" % (out_label, "on" if laser==0 else "off", delay), rixs_map[laser])

    #np.savetxt("%s_map_%dps_energies.txt" % (out_label, delay), sorted(events_per_energy[0].keys()))

    return rixs_map, rixs_map_std, total_results


def get_rixs_spectra(dataset_name, runs, thr_low=0, thr_hi=999999, selection=""):
    df, runs = get_data_df(dataset_name, runs, selection=selection)
    return compute_rixs_spectra(dataset_name, df, thr_low, thr_hi)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "USAGE: %s initial_run_number final_run_number" % sys.argv[0]
    else:
        run_i = int(sys.argv[1])
        run_f = int(sys.argv[2])
        runs = [str(x) for x in np.arange(run_i, run_f + 1)]

        sel = "(x_shut == 1) & (x_status == 1) & (I0_up > 0.01) & (I0_down > 0.01) & (ND > -1) & (photon_mono_energy) > 9"
        rixs_map, rixs_map_std, total_results = get_rixs_spectra("detector_2d_1", runs, thr_low=70, thr_hi=170, selection=sel)

        #plt.figure()
        #plt.imshow(rixs_map[0] - rixs_map[1], aspect="auto", vmin=-10, vmax=10)
        #plt.colorbar()
        #plt.show()
