"""
Simple tool to plot control quantities (energy, I0, etc) and run numbers
It uses the output of syncdaq_get for control quantities, and the local SACLA
webpage for run info.
"""

import matplotlib.pyplot as plt
import pandas as pd
import sys

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("filename", help="Name of the .csv file with the average scan info", nargs='+')
    parser.add_argument("-o", "--outputdir", help="output directory", action="store", default=".")
    parser.add_argument("-p", "--plot", help="plot, do not save", action="store_true")
    parser.add_argument("-b", "--begin", help="beginning of the time perion, in YYYY-MM-DD HH:MM:SS format", action="store")
    parser.add_argument("-e", "--end", help="enf of the time perion, in YYYY-MM-DD HH:MM:SS format", action="store")

    args = parser.parse_args()
    if args.begin is None or args.end is None:
        print("[ERROR] -b and -e are compulsory arguments!")
        sys.exit(-1)
    
    # parser.add_argument("-d", "--daemon", help="download up to latest run number and keep polling (only applies if -l is specified)", action="store_true")

    begin = args.begin.replace("-", "%2F").replace(" ", "+").replace(":", "%3A")
    end = args.end.replace("-", "%2F").replace(" ", "+").replace(":", "%3A")

    url = "http://xqaccdaq01.daq.xfel.cntl.local/cgi-bin/storage/run.cgi?from_time=%s&to_time=%s&search_key=time&admin=&bl=3&mode=Search" % (begin, end)
    
    rdf = pd.read_html(url, header=0)[1]
    rdf = rdf.set_index("run #")


    fig = plt.figure(figsize=(20, 15))

    fnames = args.filename
    print(fnames)
    #fnames.append(args.filename)
    #fnames.append("/xdaq/work/share/milne/drive/Escan_0002.txt")

    for i, fname in enumerate(fnames):
        ax = plt.subplot(len(fnames), 1, i)
        #df = pd.read_csv(fname, header=0, sep=",")
        colname = fname.split("/")[-1].split(".")[0]
        ax.set_title(colname)

        df = pd.read_csv(fname, names=["StartTag", colname], sep=",")
        df = df.set_index("StartTag")

        #print df
        #q = df["xfel_bl_3_st_4_pd_user_9_fitting_peak/voltage"] / df["xfel_bl_3_st_4_pd_user_7_fitting_peak/voltage + xfel_bl_3_st_4_pd_user_8_fitting_peak/voltage"]
        #df["xfel_bl_3_st_4_pd_user_9_fitting_peak/voltage / [Ch4]"].plot(label="m9 / ch4")
        #q.plot(label="m9 / (m7 + m8)")

        for idx in rdf.index.tolist():
            t = rdf["start trigger"][idx]
            if t < df.index[0]:
                continue
            if t > df.index[-1]:
                continue
            plt.axvline(t)
            #print t,df["Mono"].mean()
            ax.text(t, df[colname].mean(), str(idx), rotation=90, fontsize=10)
        ax.plot(df.index.flatten(), df[colname].values.flatten(), 'k-')

        plt.legend(loc="best")
    if args.plot:
        plt.show()
    plt.savefig(args.outputdir + "/" + fname.split("/")[-1].replace(".txt", ".png"))
