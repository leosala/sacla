"""

Simple tool to convert SACLA hdf5 files into more analysis-friendly hdf5 files.
It also adds information gathered using syncdaq_get

"""

import h5py
import numpy as np
import sys
import os
from time import time
import pandas as pd

# Converting DAQ quantities
sys.path.append( os.environ["PWD"]+"/../" )
#import utilities as ut
import utilities.beamtime_converter_201406XX as btc

if len(sys.argv) != 3:
    print "USAGE: ", sys.argv[0], "infile.h5 outfile.h5"
    exit(-1)

start_t = time()

INFILE = sys.argv[1]
OUTFILE = sys.argv[2]

# MPCCD can be detector_2d_1 or 9, depending if Octal was in or not...
SELECT_DETECTORS = ["MPCCD-1N0-M01-001"]

DET_INFO_DSET = "/detector_info/detector_name"
RUN_INFO_DST = ["event_info", "exp_info", "run_info"]
TAG_DST = "tag"


def convert_sacla_file(f, fout, compress=""):
    """
    Converts SACLA data format in an analysis-friendly format (single dataset)
    """

    file_info = f["file_info"]
    f.copy(file_info, fout)

    # getting the run list
    run_list = []
    for k in f.keys():
        if k[0:3] == "run":
            run_list.append(k)

    print "Run list:", run_list

    for run in run_list:
        run_dst = f[run]
        fout.create_group(run)

        # list of all the tags in the file.
        # Warning! Not all the tags have data...
        print run_dst["event_info"].keys()
        tag_list = run_dst["event_info/tag_number_list"]
        print "#tags:", tag_list.shape

        # Copying Run information datasets
        for info_dst in RUN_INFO_DST:
            info = run_dst[info_dst]
            f.copy(info, fout[run])

        # Loop on detectors
        detectors_list = []
        for t in run_dst.keys():
            if t.find("detector") == -1:
                continue
            print t + DET_INFO_DSET, run_dst[t + DET_INFO_DSET].value
            if run_dst[t + DET_INFO_DSET].value in SELECT_DETECTORS:
                print "selecting", run_dst[t + DET_INFO_DSET].value, "(known as " + t + ")"
                detectors_list.append(t)

        for det in detectors_list:
            print det

            # do not forget about detector info...
            grp = run_dst[det]
            real_det_name = run_dst[det + DET_INFO_DSET].value
            tags_images = run_dst[det].keys()
            tags_images.remove('detector_info')
            tags_n = tag_list.shape[0]
            print "Total tags:", tags_n

            tags = np.zeros([tags_n], dtype="i8")
            temp = np.zeros([tags_n], dtype="float")
            status = np.zeros([tags_n], dtype="i4")
            is_data = np.zeros([tags_n], dtype="bool")

            temp[:] = np.NAN
            status[:] = np.NAN

            info = grp["detector_info"]
            print info, "/" + run + "/" + real_det_name
            fout.create_group(run + "/" + real_det_name)
            f.copy(info, fout["/" + run + "/" + real_det_name])

            data_dset = None

            chk_size = 300
            print det + "/" + tags_images[0] + "/detector_data"
            print run_dst[det + "/" + tags_images[0]].keys()
            data_shape = run_dst[det + "/" + tags_images[0] + "/detector_data"].shape
            data_type = run_dst[det + "/" + tags_images[0] + "/detector_data"].dtype

            chunk_size = (data_shape[0] / 4, data_shape[1] / 4)
            if compress != "":
                data_dset = fout.create_dataset(run + "/" + real_det_name + "/" + "detector_data", (tags_n, ) + data_shape, maxshape=(None, ) + data_shape, dtype = data_type, chunks=(1, ) + chunk_size, compression = compress, shuffle=True)
            else:
                data_dset = fout.create_dataset(run + "/" + real_det_name + "/" + "detector_data", (tags_n, ) + data_shape, maxshape=(None, ) + data_shape, dtype = data_type, chunks=(1, ) + chunk_size, )

            for chk_i in xrange(0, tag_list.size, chk_size):
                # tag_i = tag_list[chk_i]
                end = chk_i + min(chk_size, tags_n - chk_i)
                data_chk = np.zeros((min(chk_size, tags_n - chk_i),) + data_shape, dtype=data_type)
                data_chk[:] = np.NAN

                for c in xrange(min(chk_size, tags_n - chk_i)):
                    tag_str = "tag_" + str(tag_list[c + chk_i])  # "tag_" + str(tag_i + c)
                    if (tag_str) in tags_images:
                        is_data[chk_i + c] = True
                        data_chk[c] = run_dst[det + "/" + tag_str + "/detector_data"][:]
                        tags[chk_i + c] = int(tag_str[4:].strip())
                        temp[chk_i + c] = run_dst[det + "/" + tag_str + "/temperature"].value
                        status[chk_i + c] = run_dst[det + "/" + tag_str + "/detector_status"].value

                # print chk_i, chk_i + chk_size, end, data_chk[0, 0, 0]
                data_dset[chk_i:end] = data_chk[:]

            tags_dset = fout.create_dataset(run + "/" + real_det_name + "/" + TAG_DST, data=tags)
            temp_dset = fout.create_dataset(run + "/" + real_det_name + "/temperature", data=temp)
            status_dset = fout.create_dataset(run + "/" + real_det_name + "/detector_status", data=status)
            isdata_dset = fout.create_dataset(run + "/" + real_det_name + "/is_data", data=is_data)

    #f.close()
    #fout.close()

    print "Total time: ", time() - start_t, "s"


if __name__ == "__main__":
    print INFILE

    if INFILE != OUTFILE:
        from shutil import copyfile
        copyfile(INFILE, OUTFILE)
    #sys.exit()

    f = h5py.File(INFILE, "r")
    fout = h5py.File(OUTFILE, "a")
    #add_files_dir = "/home/sala/Work/Data/Sacla/DAQ/timbvd/"
    #add_files_dir = "/media/sala/Elements/Data/Sacla/DAQ/timbvd/"
    add_files_dir = "/swissfel/photonics/data/2014-06-11_SACLA_ES2/DAQ/timbvd/"

    # this step can be avoided, if you want to keep original SACLA data file structure
    #convert_sacla_file(f, fout)
    #convert_sacla_file(f, fout, compress="lzf")

    

    # Add information from syncdaq_get into the HDF5 file produced by DataConvert3
    daq_info = {}
    daq_info["delay"] = {"fname": "Delays.txt", "units": "ps"}
    daq_info["energy"] = {"fname": "Mono.txt", "units": "eV"}
    daq_info["x_shut"] = {"fname": "Xshut.txt", "units": "bool"}
    daq_info["x_status"] = {"fname": "Xstat.txt", "units": "bool"}
    daq_info["laser_on"] = {"fname": "LaserOn.txt", "units": "bool"}
    daq_info["bl2_I0mon_up"] = {"fname": "X2Up.txt", "units": "V"}
    daq_info["bl2_I0mon_down"] = {"fname": "X2Down.txt", "units": "V"}
    daq_info["bl2_I0mon_right"] = {"fname": "X2Right.txt", "units": "V"}
    daq_info["bl2_I0mon_left"] = {"fname": "X2Left.txt", "units": "V"}
    daq_info["johann_apd"] = {"fname": "APD.txt", "units": "V"}
    daq_info["johann_theta"] = {"fname": "Johann.txt", "units": "pulse"}
    daq_info["johann_trans"] = {"fname": "APD_trans.txt", "units": "V"}

    daq_info["Trans_PD2"] = {"fname": "PD.txt", "units": "V"}
    daq_info["TFY_PD9"] = {"fname": "PD9.txt", "units": "V"}
    daq_info["I0up_PD3"] = {"fname": "X41.txt", "units": "V"}
    daq_info["I0down_PD4"] = {"fname": "X42.txt", "units": "V"}
    daq_info["laserpos_h_M27"] = {"fname": "M27.txt", "units": "pulse"}
    daq_info["laserpos_v_M28"] = {"fname": "M28.txt", "units": "pulse"}

    run_list = []
    for k in f.keys():
        if k[0:3] == "run":
            run_list.append(k)

    for dname, v in daq_info.iteritems():
        df = pd.read_csv(add_files_dir + v["fname"], header=0, names=["tag", "value"], index_col="tag", )

        for r in run_list:
            tags = f[str(r) + "/event_info/tag_number_list"]
            red_df = df.loc[tags[0]:tags[-1]]
            conv_df = btc.convert(dname, red_df[:])
            dt = np.float
            if v["units"] == "bool" or v["units"] == "pulse":
                dt = np.int
            tags_dset = fout.create_dataset(str(r) + "/daq_info/" + dname, data=conv_df, chunks=True, dtype=dt)
            tags_dset.attrs["units"] = np.string_(v["units"])
            # do a write here???
            fout.flush()

    f.close()
    fout.close()
