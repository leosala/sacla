from plot_xas import compute_xas
import matplotlib.pyplot as plt


scan_type = "delay"
run_list = [
    [257702, 257783],
    [258125, 258201],
    [258202, 258278],
    [258607, 258634],
    [258712, 258736]
    ]
for runs in run_list:
    print runs
    t0 = 221
    #if runs[0] > 258202:
    #    t0 = 0
    df, df_conds = compute_xas(scan_type, runs[0], runs[1], t0=t0)
    df_on = df[df["laser"] == 1].mean(level=0)
    df_off = df[df["laser"] == 0].mean(level=0)
    df_diff = (df_on - df_off)

    #df_on_std = df[df["laser"] == 1].std(level=0)
    #df_off_std = df[df["laser"] == 0].std(level=0)
    #df_diff_std = np.sqrt((df_on_std) * (df_on_std) + (df_off_std) * (df_off_std))

    plt.plot(df_diff.index, df_diff["absorp"], label="%d - %d" % (runs[0], runs[1]))

plt.legend(loc="best")
plt.show()
