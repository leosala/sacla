
from plot_xas import compute_xas
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

labels = []
### Various run ranges
# EXAFS: 
#     [259001, 259038], [259038, 259057], [259058, 259075], [259076, 259095], [259096, 259113], [259114, 259132],  [259133, 259152], [259153, 259170], [259171, 259190], [259191, 259208],[259209, 259228]
# t0 checks:
#    [257702, 257783],
#    [258125, 258201],
#    [258202, 258278],
##    [258607, 258634],
#    [258712, 258736],
#    [259229, 259400]
# 258279 (5 ps), 258331 (10 ps) and 258385 (50 ps) 258554 (5 ps)

# Chris check
#    [258279, 258330], [258331, 258385], [258386, 258439], [258554, 258606]
#labels = ["5 ps", "10 ps", "50 ps", "5 ps"]

### Can be delay, energy
scan_type = "energy"
run_list = [
#    [257702, 257783],
#   [258125, 258201],
#   [258202, 258278],
##    [258607, 258634],
#    [258712, 258736],
#    [259229, 259286],
#    [259305, 259376],
    [259379, 259428],
    [259429, 259482],
    [259483, 259500],
    ]

labels = ["2 ps", "1 ps", "2 ps"]
fig = plt.figure(figsize=(20, 15))
a2 = plt.subplot(111)

markers = ["o", "v", "^", ">", "<", "1", "2", "3", "4", "*", "s", "p", "h", "H", "+", "x"]
for i, runs in enumerate(run_list):
    print runs

    labels.append("")
    # delay time 0
    t0 = 221

    # Getting data from HDF5:
    df, df_conds, end_run = compute_xas(scan_type, runs[0], runs[1], t0=t0)

    # Dividing laser on / off, and averaging
    df_on = df[df["laser"] == 1].mean(level=0)
    df_off = df[df["laser"] == 0].mean(level=0)
    # on - off
    df_diff = 10 * (df_on - df_off)
    # Std deviation
    df_on_std = df[df["laser"] == 1].std(level=0)
    df_off_std = df[df["laser"] == 0].std(level=0)
    df_diff_std = np.sqrt((df_on_std) * (df_on_std) + (df_off_std) * (df_off_std))

    # Plotting
    df_diff["absorp"].plot(label="%d - %d  %s 10xdiff" % (runs[0], int(end_run), labels[i]), linewidth=2., marker=markers[i])
    #a2.fill_between(df_diff.index.tolist(), (df_diff - df_diff_std)["absorp"].values.tolist(), (df_diff + df_diff_std)["absorp"].values.tolist(), alpha=0.3, )  #  edgecolor='#999999', facecolor='#999999')
    df_on["absorp"].plot(label="%d - %d on" % (runs[0], runs[1]), linewidth=2., marker=markers[i])
    df_off["absorp"].plot(label="of %d - %d" % (runs[0], runs[1]), linewidth=2.)
    # plt.plot(df_on.index, df_on["absorp"], label="on %d - %d" % (runs[0], runs[1]))
    # plt.plot(df_off.index, df_off["absorp"], label="off %d - %d" % (runs[0], runs[1]))
    # plt.plot(df_diff.index, 10 * df_diff["absorp"], label="10x diff %d - %d" % (runs[0], runs[1]))


plt.legend(loc="best")
plt.show()
