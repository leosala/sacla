"""

Simple tool to convert SACLA hdf5 files into more analysis-friendly hdf5 files.

"""

import h5py
import numpy as np
from sys import argv, exit
from time import time
import pandas as pd

if len(argv) != 3:
    print "USAGE: ", argv[0], "infile.h5 outfile.h5"
    exit(-1)

start_t = time()

INFILE = argv[1]
OUTFILE = argv[2]

SELECT_DETECTORS = ["detector_2d_9"]

RUN_INFO_DST = ["event_info", "exp_info", "run_info"]
TAG_DST = "tag"

# max items per loop
MAX_ITEMS = 100


def convert_sacla_file(f, fout):
    """
    Converts SACLA data format in an analysis-friendly format
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

        # Copying Run information datasets
        for info_dst in RUN_INFO_DST:
            info = run_dst[info_dst]
            f.copy(info, fout[run])

        # Loop on detectors
        detectors_list = []
        for t in run_dst.keys():
            if t in SELECT_DETECTORS:
                if t.find("detector") == -1:
                    continue
                detectors_list.append(t)

        for det in detectors_list:
            print det
            # do not forget about detector info...
            grp = run_dst[det]
            tags_n = len(grp.items()) - 1  # TODO dirty trick to get rid of detector_info quickly...
            tags = np.zeros([tags_n], dtype="i8")
            temp = np.zeros([tags_n], dtype="float")
            status = np.zeros([tags_n], dtype="i4")

            info = grp["detector_info"]
            print info, "/" + run + "/" + det
            fout.create_group(run + "/" + det)
            f.copy(info, fout["/" + run + "/" + det])

            data = None
            data_dset = None
            first_flag = True

            i = 0
            data_i = 0
            for tag_str in run_dst[det].keys():
                if tag_str == "detector_info":
                    continue

                if first_flag:
                    data_shape = run_dst[det + "/" + tag_str + "/detector_data"].shape
                    data_type = run_dst[det + "/" + tag_str + "/detector_data"].dtype
                    print data_shape, data_type
                    chunk_size = (data_shape[0] / 4, data_shape[1] / 4)

                    data_dset = fout.create_dataset(run + "/" + det + "/" + "detector_data", (tags_n, ) + data_shape, maxshape=(None, ) + data_shape, dtype = data_type, chunks=(1, ) + chunk_size)  # compression = "lzf", shuffle=True,

                    data = np.zeros((MAX_ITEMS,) + data_shape, dtype=data_type)
                    first_flag = False

                tags[i] = int(tag_str[4:].strip())
                temp[i] = run_dst[det + "/" + tag_str + "/temperature"].value
                temp[i] = run_dst[det + "/" + tag_str + "/detector_status"].value

                if data_i % MAX_ITEMS == 0 and data_i != 0:
                    data_dset[i - MAX_ITEMS: i] = data
                    data = np.zeros((MAX_ITEMS,) + data_shape, dtype=data_type)
                    data_i = 0

                data[data_i, :, :] = run_dst[det + "/" + tag_str + "/detector_data"][:]

                i += 1
                data_i += 1

            if data_i <= MAX_ITEMS:
                print "Writing last ", data_i, "elements"
                data_dset[i - MAX_ITEMS: i] = data

            tags_dset = fout.create_dataset(run + "/" + det + "/" + TAG_DST, data=tags)
            temp_dset = fout.create_dataset(run + "/" + det + "/temperature", data=temp)
            status_dset = fout.create_dataset(run + "/" + det + "/detector_status", data=status)

    #f.close()
    #fout.close()

    print "Total time: ", time() - start_t, "s"


if __name__ == "__main__":
    f = h5py.File(INFILE)
    fout = h5py.File(OUTFILE, "w")
    add_files_dir = "/home/sala/Work/Data/Sacla/DAQ/timbvd/"
    add_files = ["Delays.txt"]

    convert_func = lambda x : x.replace("pulse", "")

    for af in add_files:
        if af == "Delays.txt":
            convert_func = lambda x : float(x.replace("pulse", "")) / 149.902
        df = pd.read_csv(add_files_dir + af, header=0, names=["tag", "value"], index_col="tag", converters={"value": convert_func})
        print df

    run_list = []
    print f.keys()
    for k in f.keys():
        if k[0:3] == "run":
            run_list.append(k)
    for r in run_list:
        tags = f[str(r) + "/event_info/tag_number_list"]
        red_df = df.loc[tags[0]:tags[-1]]

    #convert_sacla_file(f, fout)

