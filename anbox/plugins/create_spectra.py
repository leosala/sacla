import h5py
import numpy as np
import os
import sys
import multiprocessing as mproc
from operator import add

try:
    import utilities as ut
except:
    # loading some utils
    sys.path.append(os.path.split(__file__)[0] + "/../../")
    import utilities as ut


def analysis_worker(pout, an_func, fname, dst_name, tags_list, masks, thr):
    h5file = h5py.File(fname, 'r')
    h5_dst = h5file[dst_name]
    ret = an_func(h5_dst, tags_list, masks=masks, thr=thr)
    h5file.close()
    pout.send(ret)


class CreateSpectraPumpProbe(object):

    def apply(self, fname):
        self.fname = fname
        self.h5file = h5py.File(self.fname, 'r')
        self.run_number = self.fname.split("/")[-1].split(".")[0]
        self.tags_list = self.h5file["/run_" + self.run_number + "/event_info/tag_number_list"][:]
        self.h5_dst = self.h5file["/run_" + self.run_number + "/detector_2d_1/"]
        is_laser = self.h5file["/run_" + self.run_number + "/event_info/bl_3/lh_1/laser_pulse_selector_status"][:]

        self.is_laser_off = is_laser == 0
        self.is_laser_on = is_laser == 1
        self.h5file.close()

    def run(self, njobs=4):
        processes = []
        splitted_tags_list = np.array_split(self.tags_list, njobs)
        splitted_mask = np.array_split(self.masks, njobs)
        print splitted_tags_list[0].shape
        for j in range(njobs):
            # print "started job", j
            an_func = ut.analysis.get_spectrum_sacla  # corr=corr, apply_corr=apply_corr, roi=roi, masks=[[is_laser_on], [is_laser_off]])

            pin, pout = mproc.Pipe()
            p = mproc.Process(target=analysis_worker, args=(pout, an_func, self.fname, self.dst_name, splitted_tags_list[j], splitted_mask[j], self.thr))
            p.start()
            processes.append((p, pin, pout))

        ret = None
        for j in range(njobs):
            ret_t = processes[j][1].recv()
            print sum(ret_t[1])
            if ret is None:
                ret = ret_t
            else:
                for i in range(len(ret)):
                    ret[i] = ret[i] + ret_t[i]
            processes[j][0].join()

        return ret
