import h5py
import sys
import os
# loading some utils
sys.path.append(os.environ["PWD"] + "/../")
from utilities import sacla_hdf5

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('run', metavar='run', type=int, nargs=1, help='an integer for the accumulator')
    parser.add_argument("-i", "--indir", help="""Directory where input files are stored. Default: .""", action="store", default=".")
    parser.add_argument("-o", "--outdir", help="""Directory where output files are stored. Default: .""", action="store", default=".")
    args = parser.parse_args()
    print args

    roi = [[0, 1024], [325, 335]]  # X, Y

    run = str(args.run[0])
    hdf5FileName = args.indir + '/' + run + '.h5'
    hdf5FileName_ROI = args.outdir + '/' + run + '_roi.h5'

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
    #runs = sacla_hdf5.get_run_metadata(f)
    #metadata = sacla_hdf5.get_metadata(runs, variables)
    #sacla_hdf5.write_metadata(hdf5FileName_ROI, metadata)
    f_out = h5py.File(hdf5FileName_ROI, 'a')
    run_dst = f["/run_" + run]

    detector_names = ["MPCCD-1N0-M01-001"]
    detectors_list = []
    detector_dstnames = [i for i in run_dst.keys() if i.find("detector_2d") != -1]
    for d in detector_dstnames:
        print run_dst[d + "/detector_info/detector_name"].value
        if run_dst[d + "/detector_info/detector_name"].value in detector_names:
            detectors_list.append(d)
    #run_243561/detector_2d_assembled_2/detector_info/detector_name
    print detectors_list
    tag_list = f["/run_" + run + "/event_info/tag_number_list"][:]
    DET_INFO_DSET = "/detector_info/detector_name"
    RUN_INFO_DST = ["event_info", "exp_info", "run_info"]
    file_info = f["file_info"]
    f.copy(file_info, f_out)
    f_out.create_group("/run_" + run)

    for info_dst in RUN_INFO_DST:
        info = run_dst[info_dst]
        f.copy(info, f_out["/run_" + run])

    for dreal in detectors_list:
        detector_dsetname = "/run_" + run + "/" + dreal
        grp = f_out.create_group(detector_dsetname)

        sacla_hdf5.get_roi_data(f[detector_dsetname], f_out[detector_dsetname], tag_list, roi)
    f_out.close()
