# # Sacla - RIXS example
# 
# In this notebook, some simple code is provided as an example to produce some Resonant Inelastic X-Ray Scattering (RIXS) maps.
# 
# The basic steps to be performed are:
# - select a suitable set of runs
# - scan for finding the scanned monochromator energies
# - produce the spectra for each energy, and the on/off maps
# 
# *The aim of this tutorial is not to produce extremely efficient code, but code as simple and as fast as possible to support the data quality evaluation during beamtime and after.*
# 
# We'll start with the usual set of imports:

import numpy as np
import matplotlib as mpl
# loading customized matplotlib style. If not available, it does nothing
#try:
#    mpl.rcParams = mpl.rc_params_from_file("/swissfel/photonics/sala/sacla/utilities/matplotlibrc")
#except:
#    pass

import matplotlib.pyplot as plt
import sys
import h5py
import pandas as pd

# Loading SACLA tools 
SACLA_LIB = "../"
sys.path.append(SACLA_LIB)
import utilities as ut

# specific converters for the 2014-11 data taking. These should be customized per each beamtime!
from utilities import beamtime_converter_201411XX as sacla_converter

# directory containing ROI'd hdf5 files
DIR = "/swissfel/photonics/data/2014-11-26_SACLA_ZnO/hdf5/"

# you want the final plot?
plot_figs = True

# runs to be analyzed
runs = [str(x) for x in range(258874, 258884)]
# label for ascii output dump
out_label = "rixs_" + runs[0] + "-" + runs[-1]

# set the dataset, and the ADU threshold
dataset_name = "detector_2d_1"
thr = 70


# Then, we define:
# * the SACLA datasets
# * $t_0$
# * the runs to be analyzed

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


# In principle, a single run can contain *multiple mono settings*, so we need to load data from all the runs, and the group them by mono energy. `Pandas` can help us with that...
# 
# We load all data from files, place it in a `DataFrame`, and then add some useful derived quantities. At last, we use `tags` as index for the `DataFrame`

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

# The last preliminary step is to filter out garbage data. As a bonus, you can also find out at which `tag` the mono scan setting changed:
# preparing the is_data mask
is_data = (df_orig.x_shut == 1) & (df_orig.x_status == 1) & (df_orig.photon_mono_energy > 9.6)
is_data = is_data & (df_orig.I0_up > 0.01) & (df_orig.I0_down > 0.01) & (df_orig.ND > -1)

# filtering out garbage
df = df_orig[is_data]

# print selection efficiency
print "\nSelection efficiency"
sel_eff = pd.DataFrame( {"Total":df_orig.groupby("run").count().ND, 
                         "Selected": df.groupby("run").count().ND, 
                         "Eff.": df.groupby("run").count().ND / df_orig.groupby("run").count().ND})
print sel_eff

# checking delay settings
g = df.groupby(['run', 'delay'])
print "\nEvents per run and delay settings"
print g.count().TFY

delay = df.delay.unique()
if len(delay) > 1:
    print "More than one delay settings in the selected run range, exiting"
    sys.exit(-1)


print "\nAvailable energy settings"
print df.photon_mono_energy.unique()
print ""

# getting quantities when a variable changes
#tmp = df.photon_mono_energy.values[1:] - df.photon_mono_energy.values[0:-1]
#mask =tmp!=0
#mask = np.insert(mask, 0, True, )
# this is where it changes
#print df[mask].index.tolist()


# Now we can run the analysis. For each energy value and each run, a *list of tags* is created, such that events hage the same mono energy and they are part of the same run (as each run is in a separated file). For each of these lists, we run the `AnalysisProcessor` and create the required spectra, for laser on and off. 

# the mono energies contained in the files
energies_list = sorted(df.photon_mono_energy.unique().tolist())

# The AnalysisProcessor
an = ut.analysis.AnalysisProcessor()
# if you want a flat dict as a result
an.flatten_results = True
    
# add analysis
an.add_analysis("image_get_spectra", args={'axis': 1, })  #'thr_low': thr,})
an.add_analysis("image_get_mean_std", )  #args={'thr_low': thr})
#bins = np.arange(-150, 300, 5)
#an.add_analysis("image_get_histo_adu", args={'bins': bins})

an.set_sacla_dataset(dataset_name)

# add preprocess steps
an.add_preprocess("image_set_thr", args={'thr_low': thr})
        
# run the analysis
n_events = -1
spectrum_on = None
spectrum_off = None

fnames = [DIR + str(run) +"_roi.h5" for run in runs]


# To save some time, we can try a trivial parallelization on the analysis process, using the `multiprocessing` module. On a busy multicore system, this should take ~2:00 (compared to ~5:00 with the simple loop) minutes with 4 or more parallel jobs.  
# One difference w.t.r. of the simple loop is that in this case we are calling `an`, and not `an.analyze_images`: this is because of some technicalities of the `multiprocessing` module regarding pickling and unpickling, which are of no interest here. Just keep in mind that *for an AnalysisProcessor object calling* `an()` *or* `an.analyze_images()` *is the same*.

from multiprocessing import Pool
from multiprocessing.pool import ApplyResult

# initialization of the RIXS maps
rixs_map_on = np.zeros((len(energies_list), 1024))
rixs_map_off = np.zeros((len(energies_list), 1024))
rixs_map_on_std = np.zeros((len(energies_list), 1024))
rixs_map_off_std = np.zeros((len(energies_list), 1024))

n_events = -1
spectrum_on = None
spectrum_off = None

for i, energy in enumerate(energies_list):
    async_results = []  # list for results

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
        
        if not results[j].has_key("spectra"):
            continue

        energy_mask = energy_masks[j]
        laseron_mask = energy_mask.is_laser.values[:n_events]
        spectrum_on = np.nansum(results[j]["spectra"][laseron_mask], axis=0)
        rixs_map_on[energies_list.index(energy)] += spectrum_on
        spectrum_off = np.nansum(results[j]["spectra"][~laseron_mask], axis=0)
        rixs_map_off[energies_list.index(energy)] += spectrum_off
        
        # quadrature sums of Standard Deviations
        spectrum_on_std = np.nanstd(results[j]["spectra"][laseron_mask], axis=0)
        rixs_map_on_std[energies_list.index(energy)] += spectrum_on_std * spectrum_on_std
        spectrum_off_std = np.nanstd(results[j]["spectra"][~laseron_mask], axis=0)
        rixs_map_off_std[energies_list.index(energy)] += spectrum_off_std * spectrum_off_std

np.savetxt("%s_map_on_%dps.txt" % (out_label, delay), rixs_map_on)
np.savetxt("%s_map_off_%dps.txt" % (out_label, delay), rixs_map_off)

if plot_figs:
    #plt.figure()
    f, (ax, ax1, ax2) = plt.subplots(1, 3, sharex=True, sharey=True)
    imgplot = ax.imshow(rixs_map_off, 
                        origin="lower",
                        extent=(rixs_map_off.shape[0], rixs_map_off.shape[1], energies_list[0], energies_list[-1]),
                        aspect="auto",
                        #interpolation="nearest",
                        cmap="bwr",
                        )

    ax.get_yaxis().set_major_formatter( mpl.ticker.FuncFormatter(lambda x, p: format(float(x), ',')))
    plt.colorbar(imgplot)
    ax.set_title("RIXS Off, delay=%dps" % delay)
    ax.set_ylabel("Incoming energy (eV)")
    
    ax1 = plt.subplot(132)
    imgplot = ax1.imshow(rixs_map_on, 
                        origin="lower",
                        extent=(rixs_map_on.shape[0], rixs_map_on.shape[1], energies_list[0], energies_list[-1]),
                        aspect="auto",
                        #interpolation="nearest",
                        cmap="bwr",
                        )

    ax1.get_yaxis().set_major_formatter( mpl.ticker.FuncFormatter(lambda x, p: format(float(x), ',')))
    plt.colorbar(imgplot)
    plt.title("RIXS On, delay=%dps" % delay)
    plt.ylabel("Incoming energy (eV)")

    ax2 = plt.subplot(133)
    rixs_map = rixs_map_on - rixs_map_off
    imgplot = ax2.imshow(rixs_map, 
                        origin="lower",
                        extent=(rixs_map.shape[0], rixs_map.shape[1], energies_list[0], energies_list[-1]),
                        aspect="auto",
                        #interpolation="nearest",
                        cmap="bwr",
                        )

    ax2.get_yaxis().set_major_formatter( mpl.ticker.FuncFormatter(lambda x, p: format(float(x), ',')))
    plt.colorbar(imgplot)
    plt.title("RIXS Off, delay=%dps" % delay)
    plt.ylabel("Incoming energy (eV)")
    

    plt.tight_layout()
    plt.show()
