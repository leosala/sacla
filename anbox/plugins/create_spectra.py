import h5py
import numpy as np
import os
import sys

try:
    import utilities as ut
except:
    # loading some utils
    sys.path.append(os.path.split(__file__)[0] + "/../../")
    import utilities as ut


class CreateSpectraPumpProbe(object):

    def apply(self, fname):
        self.fname = fname
        print self.fname
        self.h5file = h5py.File(self.fname, 'r')
        run = self.fname.split("/")[-1].strip(".h5")
        self.tags_list = self.h5file["/run_" + run + "/event_info/tag_number_list"][:]
        #print self.h5file["/run_" + run + "/detector_2d_1"].keys()[0:10]
        #print self.h5file["/run_" + run + "/detector_2d_1/tag_" + str(self.tags_list[0])].keys()
        self.h5_dst = self.h5file["/run_" + run + "/detector_2d_1/"]
        is_laser = f["/run_" + run + "/event_info/bl_3/lh_1/laser_pulse_selector_status"][:]

        self.is_laser_off = is_laser == 0
        self.is_laser_on = is_laser == 1

    def run(self):
        ut.cython_utils.get_spectrum_sacla(self.h5_dst, self.tags_list, )  # corr=corr, apply_corr=apply_corr, roi=roi, masks=[[is_laser_on], [is_laser_off]])
        return 0
