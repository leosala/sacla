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

import json
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

s = "25/11/2014 11:59"
e = "26/11/2014 12:00"


class UpdateDAQ(object):
    def __init__(self, ax, start_time, stop_time, daq_quantities, plot=None, json_name=None, csv_name=None):
        self.start_time = start_time
        self.stop_time = stop_time
        self.data = None
        self.daq_quantities = daq_quantities
        self.plot = plot
        self.json_name = json_name
        self.csv_name = csv_name
        
        self.line, = ax.plot([], [], linestyle='', marker="*")
        self.ax = ax

        # Set up plot parameters
        self.ax.grid(True)

    def init(self):
        self.line.set_data([], [])
        return self.line,

    def update_data(self):
        old_stop = self.stop_time

        self.start_time = old_stop
        # TODO change this with NOW
        #self.stop_time = old_stop + 60
        self.stop_time = time()

        data_on_tmp = sacla_hdf5.get_daq_data(self.daq_quantities, start_time=self.start_time, stop_time=self.stop_time, cond="xfel_bl_3_shutter_1_open_valid/status = 1")

        if self.data is not None:
            for k in daq_quantities.keys():
                self.data[k][1].extend(data_on_tmp[k][1])
        else:
            self.data = data_on_tmp.copy()

    def __call__(self, i):
        self.update_data()
        x = self.data[self.plot['x']][1]
        y = self.data[self.plot['y']][1]

        while len(x) != len(y):
            if len(x) > len(y):
                x.pop()
            else:
                y.pop()
        if self.json_name is not None:
            json_file = open(self.json_name, 'w')
            json_dict = {}
            json_dict["run"] = ""
            json_dict["name"] = "-".join(daq_quantities.keys())
            json_dict["plot_type"] = "scatter"
            if self.plot is not None:
                json_dict["label_x"] = self.plot['x']
                json_dict["label_y"] = self.plot['y']
            json_dict["data"] = []
            json_dict["data"].append(x)
            json_dict["data"].append(y)
   
            json.dump(json_dict, json_file)
   
            json_file.close()

        if self.csv_name is not None:
            import csv
            with open(self.csv_name, 'wb') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)

                spamwriter.writerow([self.plot["x"], self.plot["y"]])
                for i, x_i in enumerate(x):
                    spamwriter.writerow([x_i, y[i]])
            csvfile.close()

        if len(x) == 0 or len(y) == 0:
            return self.line,

        xmin, xmax = self.ax.get_xlim()
        dmax = max(x)
        dmin = min(x)
        ymax = max(y)
        ymin = min(y)

        self.ax.set_xlim(float(dmin), float(dmax))
        self.ax.set_ylim(0.5 * float(ymin), 1.5 * float(ymax))
        #    ax.figure.canvas.draw()

        self.line.set_data(x, y)
        return self.line,


start_time = mktime(datetime.datetime.strptime(s, "%d/%m/%Y %H:%M").timetuple())
stop_time = time()  # mktime(datetime.datetime.strptime(e, "%d/%m/%Y %H:%M").timetuple())

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ud = UpdateDAQ(ax, start_time=start_time, stop_time=stop_time, daq_quantities=daq_quantities, plot={'x': 'I0', 'y': 'Mono'}, json_name="daq.json", csv_name="daq.csv")
anim = FuncAnimation(fig, ud, interval=1000, blit=True)
plt.show()


