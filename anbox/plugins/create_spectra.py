import h5py
import numpy as np
import os
import sys
import multiprocessing as mproc


try:
    import utilities as ut
except:
    # loading some utils
    sys.path.append(os.path.split(__file__)[0] + "/../../")
    import utilities as ut


def analysis_worker(an_func, fname, dst_name, tags_list):
    h5file = h5py.File(fname, 'r')
    h5_dst = h5file[dst_name]
    ret = an_func(h5_dst, tags_list)
    h5file.close()
    print "A", ret
    return ret


class CreateSpectraPumpProbe(object):

    def apply(self, fname):
        self.fname = fname
        self.h5file = h5py.File(self.fname, 'r')
        self.run_number = self.fname.split("/")[-1].split(".")[0]
        print "/run_" + self.run_number + "/event_info/tag_number_list"
        self.tags_list = self.h5file["/run_" + self.run_number + "/event_info/tag_number_list"][:]
        #print self.h5file["/run_" + run + "/detector_2d_1"].keys()[0:10]
        #print self.h5file["/run_" + run + "/detector_2d_1/tag_" + str(self.tags_list[0])].keys()
        self.h5_dst = self.h5file["/run_" + self.run_number + "/detector_2d_1/"]
        is_laser = self.h5file["/run_" + self.run_number + "/event_info/bl_3/lh_1/laser_pulse_selector_status"][:]

        self.is_laser_off = is_laser == 0
        self.is_laser_on = is_laser == 1
        self.h5file.close()

        
    def run(self, njobs=8):
        processes = []
        splitted_tags_list = np.array_split(self.tags_list, njobs)
        for j in range(njobs):
            print "started job", j
            an_func = ut.analysis.get_spectrum_sacla  # corr=corr, apply_corr=apply_corr, roi=roi, masks=[[is_laser_on], [is_laser_off]])

            pin, pout = mproc.Pipe()
            dst_name = "/run_" + self.run_number + "/detector_2d_1/"
            p = mproc.Process(target=analysis_worker, args=(an_func, self.fname, dst_name, splitted_tags_list[j]))
            p.start()
            processes.append((p, pin, pout))

        #for j in range(njobs):
        #    print processes[j][0].join()
        #    ps[k][1].send('stop')
        #    ps[k][1].close()
        #    ps[k][2].close()
        #    #ps[k][0].join()

        return 0
