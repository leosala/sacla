import h5py
import sys
import os
# loading some utils
sys.path.append(os.environ["PWD"] + "/../")
from utilities import sacla_hdf5

import line_profiler
from time import sleep

import pyinotify

import logging
logging.basicConfig(filename='tape_migration.log',
                    format="%(process)d:%(levelname)s:%(asctime)s:%(message)s",
                    level=logging.DEBUG)

# Configurables
roi = [[0, 1024], [325, 335]]  # X, Y
# June run
detector_names = ["MPCCD-1N0-M01-001"]
# "MPCCD-1-1-002", "MPCCD-1-1-004"]


def get_roi_hdf5(indir, outdir, run, roi, detector_names, pede_thr=-1):

    hdf5FileName = indir + '/' + run + '.h5'
    hdf5FileName_ROI = outdir + '/' + run + '_roi.h5'

    # variables to be read out by 'syncdaq_get' script
    variables = {
        'PD': 'xfel_bl_3_st_3_pd_2_fitting_peak/voltage',
        'PD9': 'xfel_bl_3_st_3_pd_9_fitting_peak/voltage',
        'I0': 'xfel_bl_3_st_3_pd_4_fitting_peak/voltage',
        'M27': 'xfel_bl_3_st3_motor_27/position',
        'M28': 'xfel_bl_3_st3_motor_28/position',
        'LaserOn': 'xfel_bl_lh1_shutter_1_open_valid/status',
        'LaserOff': 'xfel_bl_lh1_shutter_1_close_valid/status',
        'Delays': 'xfel_bl_3_st3_motor_25/position',
        'Mono': 'xfel_bl_3_tc_mono_1_theta/position',
        'APD': 'xfel_bl_3_st_3_pd_14_fitting_peak/voltage',
        'LasI': 'xfel_bl3_st_3_pd_4_peak/voltage',  # Extra info laser I
        'Xshut': 'xfel_bl_3_shutter_1_open_valid/status',  # X-ray on
        'Xstat': 'xfel_mon_bpm_bl3_0_3_beamstatus/summary',  # X-ray status
        'X3': 'xfel_bl_3_st2_bm_1_pd_peak/voltage',  # X-ray i 3
        'X41': 'xfel_bl_3_st_3_pd_3_fitting_peak/voltage',  # X-ray i 4
        'X42': 'xfel_bl_3_st_3_pd_4_fitting_peak/voltage',  # X-ray i 4
        'Johann': 'xfel_bl_3_st3_motor_42/position',  # Johann theta
        'APD_trans': 'xfel_bl_3_st3_motor_17/position'  # Johann det
    }

    f = h5py.File(hdf5FileName, 'r')
    runs = sacla_hdf5.get_run_metadata(f)
    metadata = sacla_hdf5.get_metadata(runs, variables)
    sacla_hdf5.write_metadata(hdf5FileName_ROI, metadata)
    f_out = h5py.File(hdf5FileName_ROI, 'a', )

    # f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
    run_dst = f["/run_" + run]

    detectors_list = []
    detector_dstnames = [i for i in run_dst.keys() if i.find("detector_2d") != -1]
    for d in detector_dstnames:
        if run_dst[d + "/detector_info/detector_name"].value in detector_names:
            detectors_list.append(d)

    tag_list = f["/run_" + run + "/event_info/tag_number_list"][:]
    DET_INFO_DSET = "/detector_info/detector_name"
    RUN_INFO_DST = ["event_info", "exp_info", "run_info"]
    file_info = f["file_info"]
    f.copy(file_info, f_out)
    try:
        f_out.create_group("/run_" + run)
    except:
        print sys.exc_info()[1]

    for info_dst in RUN_INFO_DST:
        info = run_dst[info_dst]
        f.copy(info, f_out["/run_" + run])

    for dreal in detectors_list:
        detector_dsetname = "/run_" + run + "/" + dreal

        try:
            fout_grp = f_out.create_group(detector_dsetname)
        except:
            print sys.exc_info()[1]
        info = f[detector_dsetname]["detector_info"]
        f.copy(info, f_out[detector_dsetname])

        sacla_hdf5.get_roi_data(f[detector_dsetname], f_out[detector_dsetname], tag_list, roi, pede_thr=pede_thr)
    f_out.close()
    print "Run %s done!" % str(run)


class PClose(pyinotify.ProcessEvent):
    def _run_cmd(self, run):
        """Command ran when a new file is closed"""
        try:
            get_roi_hdf5(self.indir, self.outdir, run, roi, detector_names, pede_thr=self.pede_thr)
        except:
            print "ERROR: cannot get roi for %s" % run
            print sys.exc_info()[0]

    def process_IN_CLOSE_WRITE(self, event):
        f = event.name and os.path.join(event.path, event.name) or event.path
        if event.name.find("roi.h5") != -1:
            print "skipping %s" % event.name
        else:
            run = str(event.name.split(".")[0])
            print "%s closed, waiting 5 secs to be sure..." % event.name
            sleep(5)
            self._run_cmd(run)


def auto_reduce(indir, outdir, pede_thr=-1):
    print "Starting monitoring.... please wait"
    wm = pyinotify.WatchManager()
    handler = PClose()
    handler.indir = indir
    handler.outdir = outdir
    handler.pede_thr = pede_thr

    notifier = pyinotify.Notifier(wm, default_proc_fun=handler)
    mask = pyinotify.IN_CLOSE_WRITE  # | pyinotify.IN_CREATE

    wm.add_watch(indir, mask, rec=True, auto_add=True)
    print '==> Start completed, monitoring %s (type C^c to exit)' % indir
    notifier.loop()
    print "loop stopped, exiting"


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('run', metavar='run', type=int, nargs="*", help='Run number to reduce, assuming the file is called <runnumber>.h5. If no run number is given, then the script will start watching the input directory')
    parser.add_argument("-i", "--indir", help="""Directory where input files are stored. Default: .""", action="store", default=".")
    parser.add_argument("-o", "--outdir", help="""Directory where output files are stored. Default: .""", action="store", default=".")
    parser.add_argument("-t", "--pedestal_thr", help="""Threshold to be used when computing pedestal. Default: not computed""", action="store", default=-1)
    args = parser.parse_args()

    if args.run != []:
        run = str(args.run[0])
        get_roi_hdf5(args.indir, args.outdir, run, roi, detector_names, pede_thr=float(args.pedestal_thr))
    else:
        auto_reduce(args.indir, args.outdir, pede_thr=args.pedestal_thr)
