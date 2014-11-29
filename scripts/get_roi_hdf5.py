#!/usr/bin/env python

import h5py
import sys
import os
import numpy as np
import time

# loading some utils
sys.path.append(os.environ["PWD"] + "/../")
from utilities import sacla_hdf5


# Configurables
# ROIs: one per detector. If none, please put []
rois = [[[0, 1024], [440, 512]], [[0, 1024], [460, 490]]]  # X, Y

# Detector names
detector_names = ["MPCCD-1-1-011", "MPCCD-1N0-M01-002"]
dark_dset_names = ["/dark1", "/dark2"]


def get_roi_hdf5(hdf5FileName, hdf5FileName_ROI, run, rois, detector_names, pede_thr=-1, dark_file=""):

    if rois == []:
        for d in detector_names:
            rois.append([])
    if len(rois) != len(detector_names):
        print "ERROR: please put one ROI per detector!"
        sys.exit(-1)

    f = h5py.File(hdf5FileName, 'r')

    f_out = h5py.File(hdf5FileName_ROI, 'a', )

    if dark_file != "":
        f_dark = h5py.File(dark_file, "r")
    run_dst = f["/run_%06d" % run]

    detectors_list = []
    detector_dstnames = [i for i in run_dst.keys() if i.find("detector_2d") != -1]
    for d in detector_dstnames:
        if run_dst[d + "/detector_info/detector_name"].value in detector_names:
            detectors_list.append(d)

    tag_list = f["/run_%06d/event_info/tag_number_list" % run][:]
    DET_INFO_DSET = "/detector_info/detector_name"
    RUN_INFO_DST = ["event_info", "exp_info", "run_info"]
    file_info = f["file_info"]
    f.copy(file_info, f_out)
    try:
        f_out.create_group("/run_%06d" % run)
    except:
        print sys.exc_info()[1]

    for info_dst in RUN_INFO_DST:
        info = run_dst[info_dst]
        f.copy(info, f_out["/run_%06d" % run])

    for i, dreal in enumerate(detectors_list):
        detector_dsetname = "/run_%06d/%s" % (run, dreal)
        print detector_dsetname
        try:
            fout_grp = f_out.create_group(detector_dsetname)
        except:
            print sys.exc_info()[1]
        info = f[detector_dsetname]["detector_info"]
        f.copy(info, f_out[detector_dsetname])

        if dark_file != "":
            print "With dark correction"
            sacla_hdf5.get_roi_data(f[detector_dsetname], f_out[detector_dsetname], tag_list, rois[i], pede_thr=pede_thr, dark_matrix=f_dark[dark_dset_names[i]][:])
            f_out[detector_dsetname].attrs['dark_filename'] = np.string_(dark_file.split("/")[-1])
            print np.string_(dark_file.split("/")[-1])

        else:
            sacla_hdf5.get_roi_data(f[detector_dsetname], f_out[detector_dsetname], tag_list, rois[i], pede_thr=pede_thr)
        #asciiList = [n.encode("ascii", "ignore") for n in strList]
        #f_out[detector_dsetname + "/dark_fname"] = np.string_("aaaaaaaa")
    f_out.close()
    print "Run %s done!" % str(run)


def get_roi_latest(keep_polling, input_dir, output_dir, run, rois, detector_names, pede_thr=-1, dark_file=""):

    current_run = run

    while True:
        hdf5FileName = '%s/%06d.h5' % (input_dir, current_run)
        hdf5FileName_ROI = '%s/%06d_roi.h5' % (output_dir, current_run)

        if not os.path.isfile(hdf5FileName):
            if not keep_polling:
                print "No new files to convert"
                break

            while not os.path.isfile(hdf5FileName):
                # wait for file to appear
                time.sleep(5)

        tmp_file = hdf5FileName_ROI+'.tmp'
        if not os.path.isfile(hdf5FileName_ROI) and not os.path.isfile(tmp_file):
            open(tmp_file, 'a').close()
            get_roi_hdf5(hdf5FileName, hdf5FileName_ROI, current_run, rois, detector_names, pede_thr=pede_thr, dark_file=dark_file)
            os.remove(tmp_file)
        else:
            print "ROI for run  %06d already exists. Skipping ..." % current_run

        current_run += 1
    return


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('run', metavar='run', type=int, help='Run number to reduce, assuming the file is called <runnumber>.h5. If no run number is given, then the script will start watching the input directory')
    parser.add_argument("-i", "--indir", help="""Directory where input files are stored. Default: .""", action="store", default=".")
    parser.add_argument("-o", "--outdir", help="""Directory where output files are stored. Default: .""", action="store", default=".")
    parser.add_argument("-t", "--pedestal_thr", help="""Threshold to be used when computing pedestal. Default: not computed""", action="store", default=-1)
    parser.add_argument("-d", "--dark_file", help="""File containing the dark corrections, one per detector. Default: not used""", action="store", default="")

    parser.add_argument("-l", "--latest", help="convert up to the latest run number", action="store_true")

    args = parser.parse_args()

    if not args.latest:
        hdf5FileName = '%s/%06d.h5' % (args.indir, args.run)
        hdf5FileName_ROI = '%s/%06d_roi.h5' % (args.outdir, args.run)
        get_roi_hdf5(hdf5FileName, hdf5FileName_ROI, args.run, rois, detector_names, pede_thr=float(args.pedestal_thr), dark_file=args.dark_file)
    else:
        get_roi_latest(True, args.indir, args.outdir, args.run, rois, detector_names, pede_thr=float(args.pedestal_thr), dark_file=args.dark_file)
