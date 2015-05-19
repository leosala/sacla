import h5py
import sys
import os
# loading some utils
sys.path.append( os.environ["PWD"]+"/../" )
from utilities import sacla_hdf5

if __name__ == '__main__':
    # Inputs:
    # hdf5 file - HDF5 file with the specific SACLA structure
    #hdf5FileName = '/Users/ebner/Desktop/206178.h5'
    #hdf5FileNameMetadata = '/Users/ebner/Desktop/206178_metadata.h5'
    hdf5FileName = '../206162.h5'
    hdf5FileNameMetadata = '206162_metadata.h5'

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
    sacla_hdf5.write_metadata(hdf5FileNameMetadata, metadata)
