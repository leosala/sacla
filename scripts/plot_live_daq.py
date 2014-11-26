import h5py
import sys
import os
# loading some utils
sys.path.append(os.environ["PWD"] + "/../")
from utilities import sacla_hdf5

from time import sleep, time, mktime
import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

import logging
logging.basicConfig(filename='tape_migration.log',
                    format="%(process)d:%(levelname)s:%(asctime)s:%(message)s",
                    level=logging.DEBUG)


x_data = "Mono"
y_data = "I0"
data_mask = "LaserOn"

daq_quantities = {
    'I0': 'xfel_bl_3_tc_spec_1/energy',
    #'LaserOn': 'xfel_bl_lh1_shutter_1_open_valid/status',
    'Mono': 'xfel_bl_3_st_3_pd_2_fitting_peak/voltage', #'xfel_bl_3_tc_mono_1_theta/position',
}

#start_time = time() - 2 #- 700000
#stop_time = start_time + 2 # 600

s = "02/06/2014 11:59"
e = "02/06/2014 12:00"


class UpdateDAQ(object):
    def __init__(self, ax, start_time, stop_time, daq_quantities):
        self.start_time = start_time
        self.stop_time = stop_time
        self.data_on = None
        self.data_off = None
        self.daq_quantities = daq_quantities

        self.line, = ax.plot([], [], linestyle='', marker="*")
        self.ax = ax

        # Set up plot parameters
        self.ax.grid(True)

    def init(self):
        self.line.set_data([], [])
        return self.line,

    def __call__(self, i):
        #if i == 0:
        #    return self.init()
        
        old_start = self.start_time
        old_stop = self.stop_time

        self.start_time = old_stop
        self.stop_time = old_stop + 60

        data_on_tmp = sacla_hdf5.get_daq_data(self.daq_quantities, start_time=self.start_time, stop_time=self.stop_time, cond="xfel_bl_3_shutter_1_open_valid/status = 1")

        if self.data_on is not None:
            for k in daq_quantities.keys():
                self.data_on[k][1].extend(data_on_tmp[k][1])
        else:
            self.data_on = data_on_tmp.copy()

        xmin, xmax = self.ax.get_xlim()
        dmax = max(self.data_on["I0"][1])
        dmin = min(self.data_on["I0"][1])
        ymax = max(self.data_on["Mono"][1])
        ymin = min(self.data_on["Mono"][1])
        
        self.ax.set_xlim(float(dmin), float(dmax))
        self.ax.set_ylim(0.5 * float(ymin), 1.5 * float(ymax))
        #    ax.figure.canvas.draw()
        while len(self.data_on["I0"][1]) != len(self.data_on["Mono"][1]):
            if len(self.data_on["I0"][1]) > len(self.data_on["Mono"][1]):
                self.data_on["I0"][1].pop()
            else:
                self.data_on["I0"][1].pop()
        self.line.set_data(self.data_on["I0"][1], self.data_on["Mono"][1])
        return self.line,
        
start_time = mktime(datetime.datetime.strptime(s, "%d/%m/%Y %H:%M").timetuple())
stop_time = mktime(datetime.datetime.strptime(e, "%d/%m/%Y %H:%M").timetuple())

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ud = UpdateDAQ(ax, start_time=start_time, stop_time=stop_time, daq_quantities=daq_quantities)
anim = FuncAnimation(fig, ud, interval=1000, blit=True)
plt.show()


