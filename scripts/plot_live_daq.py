#!/usr/bin/env python

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
    def __init__(self, axes, start_time, stop_time, daq_quantities, plots=None, json_name=None, csv_name=None, operation=""):
        self.start_time = start_time
        self.stop_time = stop_time
        self.data = None
        self.daq_quantities = daq_quantities
        self.plots = plots
        self.json_name = json_name
        self.csv_name = csv_name
        self.operation = operation
        self.is_first = True

        self.ax = axes[0]
        self.ax2 = axes[1]
        self.line, = self.ax.plot([], [],  linestyle='', marker="o")
        self.line2 = [None for i in range(len(plots))]
        for i in range(len(plots)):
            self.line2[i], = self.ax2.plot([], [], linestyle='', marker="o")


        # Set up plot parameters
        self.ax.grid(True)
        self.ax2.grid(True)
        self.data_list = []
        for i in range(len(plots)):
            self.data_list.append(None)

    def init(self):
        self.line.set_data([], [])
        return self.line,

    def update_data(self):
        old_stop = self.stop_time

        if not self.is_first:
            self.start_time = old_stop
        self.is_first = False
        # TODO change this with NOW
        #self.stop_time = old_stop + 60
        self.stop_time = time()

        for p_i, plot in enumerate(self.plots):
            if "cond" in plot.keys():
                data_on_tmp = sacla_hdf5.get_daq_data(self.daq_quantities, start_time=self.start_time, stop_time=self.stop_time, cond=plot['cond'])
            else:
                data_on_tmp = sacla_hdf5.get_daq_data(self.daq_quantities, start_time=self.start_time, stop_time=self.stop_time, )
            if self.data_list[p_i] is not None:
                for k in daq_quantities.keys():
                    self.data_list[p_i][k][1].extend(data_on_tmp[k][1])
            else:
                self.data_list[p_i] = data_on_tmp.copy()

    def __call__(self, i):
        xs = []
        ys = []
        dfs = []
        self.update_data()
        for p_i, plot in enumerate(self.plots):
            if self.data_list[p_i][plot['x']] is None:
                continue
            x = self.data_list[p_i][plot['x']][1]
            y = self.data_list[p_i][plot['y']][1]

            #print x, y
            while len(x) != len(y):
                if len(x) > len(y):
                    x.pop()
                else:
                    y.pop()
            dfs.append(pd.DataFrame(np.asarray([x, y]).T, columns=[plot['x'], plot['y']], ))
            dfs[p_i] = dfs[p_i].set_index(plot['x'])
            dfs[p_i] = dfs[p_i].sort_index()
            xs.append(dfs[p_i].index.values)
            ys.append(dfs[p_i].values.flatten())

        if self.operation == "subtract":
            dftot = dfs[1].subtract(dfs[0], fill_value=0)
            dftot = dftot.sort_index()

            X = dftot.index
            Y = dftot.values
        else:
            X = dfs[0].index
            Y = dfs[0].values
            #print "Operation not supported, exiting"
            #sys.exit(-1)

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

        if len(X) > 0 and len(Y) > 0:
            self.ax.set_xlim(float(min(X)), float(max(X)))
            self.ax.set_ylim(0.5 * float(min(Y)), 1.5 * float(max(Y)))

        self.line.set_data(X.tolist(), Y.tolist())

        for i, l in enumerate(self.line2):
            if dfs[i].empty:
                continue
            if i > 0:
                xmin, xmax = self.ax2.get_xlim()
                ymin, ymax = self.ax2.get_ylim()
            else:
                xmin = dfs[i].idxmin()[self.plots[i]['y']]
                xmax = dfs[i].idxmax()[self.plots[i]['y']]
                ymin = dfs[i].min()[self.plots[i]['y']]
                ymax = dfs[i].max()[self.plots[i]['y']]

            if xmin > dfs[i].idxmin()[self.plots[i]['y']]:
                xmin = dfs[i].idxmin()[self.plots[i]['y']]
            if xmax < dfs[i].idxmax()[self.plots[i]['y']]:
                xmax = dfs[i].idxmax()[self.plots[i]['y']]
            self.ax2.set_xlim(float(xmin), float(xmax))
            if ymin > dfs[i].min()[self.plots[i]['y']]:
                ymin = dfs[i].min()[self.plots[i]['y']]
            if ymax < dfs[i].max()[self.plots[i]['y']]:
                ymax = dfs[i].max()[self.plots[i]['y']]
            self.ax2.set_ylim(float(ymin), float(ymax))

            #self.ax2.set_ylim(0.5 * float(min(ys)), 1.5 * float(max(ys)))
 
            self.line2[i].set_data(xs[i], ys[i])

        return self.line, self.line2[0], self.line2[1]


if __name__ == '__main__':
    daq_quantities = {
        'I0': 'xfel_bl_3_tc_spec_1/energy',
        'LaserOn': 'xfel_bl_lh1_shutter_1_open_valid/status',
        'Mono': 'xfel_bl_3_tc_mono_1_theta/position',
        'PD': 'xfel_bl_3_st_4_pd_user_1_fitting_peak/voltage',
        'APD': 'xfel_bl_3_st_3_pd_14_fitting_peak/voltage',
    }

    s = "2014-06-04 11:59"  # "04/06/2014 11:59"
    e = "2014-06-04 12:00"  # "04/06/2014 12:00"

    start_time = time() - 3600 #mktime(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M").timetuple())
    stop_time =  time() #mktime(datetime.datetime.strptime(e, "%Y-%m-%d %H:%M").timetuple())

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ud = UpdateDAQ([ax, ax2], start_time=start_time, stop_time=stop_time, daq_quantities=daq_quantities,
                   plots=[{'x': 'I0', 'y': 'PD', 'cond': 'xfel_bl_3_shutter_1_open_valid/status = 0'},
                          {'x': 'I0', 'y': 'PD', 'cond': 'xfel_bl_3_shutter_1_open_valid/status = 1'},],
                   json_name="daq.json", csv_name="daq.csv",
                   operation="subtract"
                   )

    anim = FuncAnimation(fig, ud, interval=1000, blit=True)
    plt.show()
