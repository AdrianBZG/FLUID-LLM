from matplotlib import pyplot as plt
import numpy as np
#matplotlib.rcParams.update(matplotlib.rcParamsDefault)
# plt.rcParams['text.usetex'] = True
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif"
# })

ys = [0.251,
      0.2487,
      0.2462,
      0.2447,
      0.2443,
      0.2434,
      0.2443,
      0.2458]
ys = np.array(ys)
percent_ys = (ys - ys[0]) / ys[0] * 100
xs = np.arange(len(ys)) + 1

fig, ax1 = plt.subplots()
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
ax1.set_xlabel(r'Context length $\tau_{max}$ ', usetex=True, fontsize=15)
ax1.set_ylabel('50-RMSE', fontsize=13)
ax1.plot(xs, ys, color='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Percent change', fontsize=13)
ax2.plot(xs, percent_ys, linestyle='none')
plt.yticks(fontsize=13)

plt.tight_layout()
plt.savefig("./plots/ICL.pdf")
plt.show()
