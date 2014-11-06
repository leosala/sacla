import h5py
import utilities as ut
from time import time

run = "206162"
#run = "206178"
#run = "206183"
#DIR = "/media/sala/Elements/Data/Sacla/"
DIR = ""

f = h5py.File(DIR + run + ".h5")
f_out = h5py.File(DIR + run + "_roi.h5", "w")

grp = f_out.create_group("/run_" + run + "/detector_2d_1")
tag_list = f["/run_" + run + "/event_info/tag_number_list"][:]

init = time()
roi = [[0, 1024], [325, 335]]  # X, Y

ut.sacla_hdf5.get_roi_data(f["/run_" + run + "/detector_2d_1/"], f_out["/run_" + run + "/detector_2d_1/"], tag_list, roi)
f_out.close()
print "time:", time() - init
