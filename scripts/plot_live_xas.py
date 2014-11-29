import h5py
import sys
import os
# loading some utils
sys.path.append(os.environ["PWD"] + "/../")
from utilities import sacla_hdf5, beamtime_converter_201406XX
import numpy as np
from time import sleep, time, mktime
import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import json
import datetime
import pytz
import subprocess

daq_quantities = {
    #'I0': 'xfel_bl_3_tc_spec_1/energy',
    'LaserOn': 'xfel_bl_lh1_shutter_1_open_valid/status',
    #'Mono': 'xfel_bl_3_tc_mono_1_theta/position',
    #'PD': 'xfel_bl_3_st_4_pd_user_1_fitting_peak/voltage',
    #'APD': 'xfel_bl_3_st_3_pd_14_fitting_peak/voltage',
    'SamplePD': 'xfel_bl_3_st_4_pd_user_9_fitting_peak/voltage',
    'ScanPD': 'xfel_bl_3_st_4_pd_user_10_fitting_peak/voltage',
    'Mono': 'xfel_bl_3_tc_mono_1_theta/position',
    'Delay': 'xfel_bl_3_st_4_motor_29/position',
    'I0up': 'xfel_bl_3_st_4_pd_user_7_fitting_peak/voltage',
    'I0down': 'xfel_bl_3_st_4_pd_user_8_fitting_peak/voltage'
}

def sV(x): 
    try:
        return float(x.replace("V", ""))
    except:
        return np.nan


class UpdateDAQ(object):
    def __init__(self, axes, start_time, stop_time, json_name=None, csv_name=None, average=True):
        self.start_time = start_time
        self.stop_time = stop_time
        self.data = None
        self.daq_quantities = daq_quantities
        self.json_name = json_name
        self.csv_name = csv_name
        self.is_first = True
        self.average = average

        self.ax = axes
        self.line, = self.ax.plot([], [], linestyle='-', marker=".")
        self.line2, = self.ax.plot([], [], linestyle='-', marker=".")
        self.line3, = self.ax.plot([], [], linestyle='-', marker=".")

        # Set up plot parameters
        self.ax.grid(True)
        self.data_list = []
        self.df = [None, None]


    def update_data(self):
        old_stop = self.stop_time

        if not self.is_first:
            self.start_time = old_stop
            #self.stop_time = time()
            self.stop_time = old_stop + 60

        self.is_first = False
        # TODO change this with NOW
        
        start_time = datetime.datetime.fromtimestamp(self.start_time, )  # tz=pytz.timezone('Asia/Tokyo'))
        stop_time = datetime.datetime.fromtimestamp(self.stop_time, )  # tz=pytz.timezone('Asia/Tokyo'))

        converters = {}
        sP = lambda x : x.replace("pulse","")

        # get laseron data
        for ci, cond in enumerate(["0", "1"]):
            df = None

            for q in ["SamplePD", "I0up", "I0down"]:
                command = ['syncdaq_get', '-b', str(start_time), '-e', str(stop_time), daq_quantities[q], '-c', daq_quantities["LaserOn"] + " = " + cond, "-l", "-1", "|", "sed 's:pulse::g'", "|", "sed 's:V::g' "]
                print command
                with open(q + ".txt", "w") as outfile:
                    subprocess.call(command, stdout=outfile)
                    outfile.close()
                if df is None:
                    df = pd.read_csv(q + ".txt", names=["tag", q], sep=",", index_col=["tag"], skiprows=1, converters={q: sV})
                else:
                    df2 = pd.read_csv(q + ".txt", names=["tag", q], sep=",", index_col=["tag"], skiprows=1, converters={q: sV})
                    #df2 = df2.set_index("tag")
                    print df2[0:10]
                    df = df.join(df2)

            if self.df[ci] is None:
                self.df[ci] = df
                print self.df[ci][0:10]
            else:
                self.df[ci] = pd.concat([self.df[ci], df])

        #sys.exit(-1)


    def __call__(self, i):
        xs = []
        ys = []
        dfs = []
        self.update_data()
        for cd in range(2):
            self.df[cd]["I0"] = self.df[cd]["I0up"] + self.df[cd]["I0down"]
            print self.df[cd]

        #self.line.set_data(self.df[0].index.tolist(), self.df[0]["I0"].values.tolist())
        #self.line2.set_data(self.df[1].index.tolist(), self.df[1]["I0"].values.tolist())


        return self.line, self.line2, #self.line2[1]


if __name__ == '__main__':


    #samplepd / (ioup + iodown)

    s = "2014-11-27 20:12" #"2014-06-04 11:59"  # "04/06/2014 11:59"
    e = "2014-11-27 20:21" 	#"2014-06-04 12:00"  # "04/06/2014 12:00"

    #start_time = time() - 20000  #mktime(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M").timetuple())
    #stop_time =  time()  #mktime(datetime.datetime.strptime(e, "%Y-%m-%d %H:%M").timetuple())

    start_time = mktime(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M").timetuple())
    stop_time =  mktime(datetime.datetime.strptime(e, "%Y-%m-%d %H:%M").timetuple())

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ud = UpdateDAQ(ax, start_time=start_time, stop_time=stop_time,
                   json_name="daq.json", csv_name="daq.csv",
                   average=True
                   )

    anim = FuncAnimation(fig, ud, interval=1000, blit=True)
    plt.show()
