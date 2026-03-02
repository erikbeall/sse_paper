
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib
dsload = lambda filename: np.load(filename, allow_pickle=True, encoding='latin1').tolist()

# load datasets
[orig_radii35, orig_cmaximas35, orig_cbgs35] = np.load('results_stage1_round1_35C.npy')
[orig_radii50, orig_cmaximas50, orig_cbgs50] = np.load('results_stage1_round1_50C.npy')
[orig_radii80, orig_cmaximas80, orig_cbgs80] = np.load('results_stage1_round1_80C.npy')
orig_radii35=orig_radii35[25:]; orig_cmaximas35=orig_cmaximas35[25:]; orig_cbgs35=orig_cbgs35[25:]
pcts35 = (np.max(orig_cmaximas35)-orig_cmaximas35)/(np.max(orig_cmaximas35) - orig_cbgs35)
pcts50 = (np.max(orig_cmaximas50)-orig_cmaximas50)/(np.max(orig_cmaximas50) - orig_cbgs50)
pcts80 = (np.max(orig_cmaximas80)-orig_cmaximas80)/(np.max(orig_cmaximas80) - orig_cbgs80)

# 3 subplots, a is T_meas (maxima vs radii) for 50C dataset
# b is converted to delta 
# c is percent deviation
fig, axes = plt.subplots(1, 3, figsize=(12, 6))
c=0
# 2a is maxima of full set of 35C boson data vs radius
axes[c].plot(orig_radii35, orig_cmaximas35, 'k.')
axes[c].legend([r"$T_{T}$ = 35$^\circ$C"], fontsize=14)
axes[c].grid(True)
axes[c].set_title("a)", loc="left", fontsize=24)
axes[c].set_xlabel('Radii', fontsize=18)
axes[c].set_ylabel(r"$T_{meas}$ $^\circ$C", fontsize=18, rotation=90)
axes[c].tick_params(axis='both', which='major', labelsize=16)

# 2b is percent deviation of same
c=1
tgt35 = np.polyfit(1/np.power(orig_radii35[orig_radii35>2.75], 0.5), orig_cmaximas35[orig_radii35>2.75], 1)[1]
axes[c].plot(orig_radii35, tgt35 - orig_cmaximas35, 'k.')
axes[c].legend([r"$T_{T}$ = 35$^\circ$C"], fontsize=14)
axes[c].grid(True)
axes[c].set_title("b)", loc="left", fontsize=24)
axes[c].set_xlabel('Radii', fontsize=18)
axes[c].set_ylabel(r"$\Delta(r)$ $^\circ$C", fontsize=18, rotation=90)
axes[c].tick_params(axis='both', which='major', labelsize=16)

c=2
axes[c].plot(orig_radii35, pcts35, 'k.')
axes[c].legend([r"$T_{T}$ = 35$^\circ$C"], fontsize=14)
axes[c].set_ylim([-0.00825, 0.175])
axes[c].set_xlim([-1.5, 90])
axes[c].grid(True)
axes[c].set_title("c)", loc="left", fontsize=24)
axes[c].set_xlabel('Radii', fontsize=18)
axes[c].set_ylabel(r"Pct(r)", fontsize=18, rotation=90)
axes[c].tick_params(axis='both', which='major', labelsize=16)

#fig.tight_layout()
fig.subplots_adjust(wspace=0.5)
plt.savefig('sse_figure_3_supplement35.png')


