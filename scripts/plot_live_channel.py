import h5py
import sys
import os
# loading some utils
sys.path.append(os.environ["PWD"] + "/../")
from utilities import sacla_hdf5
import numpy as np
from time import sleep, time, mktime
import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import json
import logging
logging.basicConfig(filename='tape_migration.log',
                    format="%(process)d:%(levelname)s:%(asctime)s:%(message)s",
                    level=logging.DEBUG)
from random import randint



class UpdateDAQ(object):
    def __init__(self, axes, start_time, stop_time, daq_quantities, plot=None, json_name=None, csv_name=None, operation=""):
        self.start_time = start_time
        self.stop_time = stop_time
        self.daq_quantities = daq_quantities
        self.plot = plot
        self.json_name = json_name
        self.csv_name = csv_name
        self.operation = operation
        self.is_first = True

        self.ax = axes
        self.line, = self.ax.plot([], [],)  # linestyle='', marker=".")
        self.ax.set_ylabel(self.plot['y'])
        self.ax.set_xlabel('#tag')
        
        self.ax.grid(True)
        self.data = [[], []]

    def init(self):
        self.line.set_data([], [])
        return self.line,

    def update_data(self):
        old_stop = self.stop_time

        if not self.is_first:
            self.start_time = old_stop
            self.stop_time = time()

        self.is_first = False
        # TODO change this with NOW
        #self.stop_time = old_stop + 60

        # TODO limit sql queries
        daq_q = {}
        if self.plot['y'] in self.daq_quantities.keys():
            daq_q[self.plot['y']] = self.daq_quantities[self.plot['y']]
        if "cond" in self.plot.keys():
            data_on_tmp = sacla_hdf5.get_daq_data(daq_q, start_time=self.start_time, stop_time=self.stop_time, cond=self.plot['cond'])
        else:
            data_on_tmp = sacla_hdf5.get_daq_data(daq_q, start_time=self.start_time, stop_time=self.stop_time, )

        if self.data != []:
            self.data[1].extend(data_on_tmp[self.plot['y']][1])
            self.data[0].extend(data_on_tmp[self.plot['y']][0])
        else:
            self.data = data_on_tmp.copy()

    def __call__(self, i):
        xs = []
        ys = []
        dfs = []
        self.update_data()
        if self.data == []:
            return self.line, 
        x = self.data[0]
        y = self.data[1]

        #print x, y
        while len(x) != len(y):
            if len(x) > len(y):
                x.pop()
            else:
                y.pop()
        dfs = pd.DataFrame(np.asarray([x, y]).T, columns=["tags", self.plot['y']], )
        dfs = dfs.set_index("tags")
        dfs = dfs.sort_index()

        X = dfs.index
        Y = dfs.values
        #print "Operation not supported, exiting"
        #sys.exit(-1)

        """
        if self.json_name is not None:
            json_file = open(self.json_name, 'w')
            json_dict = {}
            json_dict["run"] = ""
            json_dict["name"] = "-".join(daq_quantities.keys())
            json_dict["plot_type"] = "scatter"
            json_dict["label_x"] = self.plots[0]['x']
            json_dict["label_y"] = self.plots[0]['y']
            json_dict["data"] = []
            json_dict["data"].append(X.tolist())
            json_dict["data"].append(Y.tolist())
            json.dump(json_dict, json_file)
            json_file.close()

        if self.csv_name is not None:
            import csv
            with open(self.csv_name, 'wb') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)

                spamwriter.writerow([self.plots[0]["x"], self.plots[0]["y"]])
                for i, x_i in enumerate(X):
                    spamwriter.writerow([x_i, Y[i]])
            csvfile.close()
        """

        if len(X) > 0 and len(Y) > 0:
            self.ax.set_xlim(dfs.idxmin()[self.plot['y']], dfs.idxmax()[self.plot['y']])
            self.ax.set_ylim(dfs.min()[self.plot['y']], dfs.max()[self.plot['y']])

        self.line.set_data(X.tolist(), Y.tolist())

        return self.line,


if __name__ == '__main__':
    daq_quantities = {
        #'I0': 'xfel_bl_3_tc_spec_1/energy',
        #'LaserOn': 'xfel_bl_lh1_shutter_1_open_valid/status',
        #'Mono': 'xfel_bl_3_tc_mono_1_theta/position',
        #'PD': 'xfel_bl_3_st_4_pd_user_1_fitting_peak/voltage',
        #'APD': 'xfel_bl_3_st_3_pd_14_fitting_peak/voltage',
        'SamplePD': 'xfel_bl_3_st_4_pd_user_9_fitting_peak/voltage',
        'Mono': 'xfel_bl_3_tc_mono_1_theta/position',
        'Delay': 'xfel_bl_3_st_4_motor_29/position'
        
    }

    s = "2014-06-27 11:20"  # "04/06/2014 11:59"
    e = "2014-06-27 12:00"  # "04/06/2014 12:00"

    #start_time = time() - 36000  # mktime(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M").timetuple())
    #stop_time =  time()  # mktime(datetime.datetime.strptime(e, "%Y-%m-%d %H:%M").timetuple())

    start_time = mktime(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M").timetuple())
    stop_time =  mktime(datetime.datetime.strptime(e, "%Y-%m-%d %H:%M").timetuple())

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ud = UpdateDAQ(ax, start_time=start_time, stop_time=stop_time, daq_quantities=daq_quantities,
                   plot={'y': 'Mono',},
                   json_name="daq.json", csv_name="daq.csv",
                   )

    anim = FuncAnimation(fig, ud, interval=5000, blit=True)
    plt.legend(loc="best")
    plt.show()
