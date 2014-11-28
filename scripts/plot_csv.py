import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("filename", help="Name of the .csv file with the average scan info", nargs='+')
    parser.add_argument("-o", "--outputdir", help="output directory", action="store", default=".")
    parser.add_argument("-p", "--plot", help="plot, do not save", action="store_true")

    #parser.add_argument("-d", "--daemon", help="download up to latest run number and keep polling (only applies if -l is specified)", action="store_true")
    url = "http://xqaccdaq01.daq.xfel.cntl.local/cgi-bin/storage/run.cgi?from_time=2014%2F11%2F27+10%3A20%3A52&to_time=2014%2F11%2F27+23%3A59%3A52&search_key=time&admin=&bl=3&mode=Search"
    rdf = pd.read_html(url, header=0)[1]
    rdf = rdf.set_index("run #")

    args = parser.parse_args()

    fig = plt.figure(figsize=(20, 15))

    fnames = args.filename
    print fnames
    #fnames.append(args.filename)
    #fnames.append("/xdaq/work/share/milne/drive/Escan_0010.txt ")

    for i, fname in enumerate(fnames):
        ax = plt.subplot(len(fnames), 1, i)
        df = pd.read_csv(fname, header=0, sep=",")
        df = df.set_index("StartTag")

        #print df
        q = df["xfel_bl_3_st_4_pd_user_9_fitting_peak/voltage"] / df["xfel_bl_3_st_4_pd_user_7_fitting_peak/voltage + xfel_bl_3_st_4_pd_user_8_fitting_peak/voltage"]
        df["xfel_bl_3_st_4_pd_user_9_fitting_peak/voltage / [Ch4]"].plot(label="m9 / ch4")
        q.plot(label="m9 / (m7 + m8)")

        plt.legend(loc="best")
    if args.plot:
        plt.show()
    plt.savefig(args.outputdir + "/" + fname.split("/")[-1].replace(".txt", ".png"))
