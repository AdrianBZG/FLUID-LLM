from matplotlib import pyplot as plt
import numpy as np
#matplotlib.rcParams.update(matplotlib.rcParamsDefault)
# plt.rcParams['text.usetex'] = True
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif"
# })

ys = [0.0565,
0.0388,
0.0322,
0.0267,
0.0251,]

ys_10 = [0.261,
0.131,
0.0974,
0.0863,
0.0892]
xs = np.arange(len(ys)) + 1

fig, ax1 = plt.subplots()
ax1.set_xticks(xs)
ax1.set_xlabel(r'Context length $\tau_{init}$ ', usetex=True, fontsize=15)
ax1.set_ylabel('1-RMSE', fontsize=13, color='tab:blue')
ax1.plot(xs, ys, color='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('10-RMSE', fontsize=13 , color='tab:red')
ax2.plot(xs, ys_10, color='tab:red')
plt.tight_layout()
plt.savefig("./plots/Fewshot.pdf")
plt.show()
