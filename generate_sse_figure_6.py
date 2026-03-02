
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
orig_radii35=orig_radii35[55:]; orig_cmaximas35=orig_cmaximas35[55:]; orig_cbgs35=orig_cbgs35[55:]
pcts35 = (np.max(orig_cmaximas35)-orig_cmaximas35)/(np.max(orig_cmaximas35) - orig_cbgs35)
pcts50 = (np.max(orig_cmaximas50)-orig_cmaximas50)/(np.max(orig_cmaximas50) - orig_cbgs50)
pcts80 = (np.max(orig_cmaximas80)-orig_cmaximas80)/(np.max(orig_cmaximas80) - orig_cbgs80)

alpha=0.725
alpha=0.5
p35=np.polyfit(1/np.power(orig_radii35, alpha), pcts35, 1)
p50=np.polyfit(1/np.power(orig_radii50, alpha), pcts50, 1)
p80=np.polyfit(1/np.power(orig_radii80, alpha), pcts80, 1)
xdata=np.linspace(1/np.power(np.min(orig_radii50), alpha), 1/np.power(np.max(orig_radii50), alpha), 100)
tgt35 = np.polyfit(1/np.power(orig_radii35[orig_radii35>2.75], alpha), orig_cmaximas35[orig_radii35>2.75], 1)[1]
tgt50 = np.polyfit(1/np.power(orig_radii50[orig_radii50>2.75], alpha), orig_cmaximas50[orig_radii50>2.75], 1)[1]
tgt80 = np.polyfit(1/np.power(orig_radii80[orig_radii80>2.75], alpha), orig_cmaximas80[orig_radii80>2.75], 1)[1]

# 6 a) is Delta for all three rgb, b) Pct for all three, c) all three fitted
fig, axes = plt.subplots(1, 3, figsize=(12, 6))
c=0
axes[c].plot(orig_radii35, tgt35 - orig_cmaximas35, 'b.')
axes[c].plot(orig_radii50, tgt50 - orig_cmaximas50, 'r.')
axes[c].plot(orig_radii80, tgt80 - orig_cmaximas80, 'g.')
axes[c].legend([r"$T_{T}$ = 35$^\circ$C",r"$T_{T}$ = 50$^\circ$C", r"$T_{T}$ = 80$^\circ$C"], fontsize=14)
axes[c].grid(True)
axes[c].set_title("a)", loc="left", fontsize=24)
axes[c].set_xlabel('Radii', fontsize=18)
axes[c].set_ylabel(r"$\Delta(r)$ $^\circ$C", fontsize=18, rotation=90)
axes[c].tick_params(axis='both', which='major', labelsize=16)

c=1
axes[c].plot(orig_radii35, pcts35, 'b.')
axes[c].plot(orig_radii50, pcts50, 'r.')
axes[c].plot(orig_radii80, pcts80, 'g.')
axes[c].legend([r"$T_{T}$ = 35$^\circ$C",r"$T_{T}$ = 50$^\circ$C", r"$T_{T}$ = 80$^\circ$C"], fontsize=14)
axes[c].grid(True)
axes[c].set_title("b)", loc="left", fontsize=24)
axes[c].set_xlabel('Radii', fontsize=18)
axes[c].set_ylabel(r"Pct(r)", fontsize=18, rotation=90)
axes[c].tick_params(axis='both', which='major', labelsize=16)

c=2
axes[c].plot(np.power(orig_radii35, -alpha), pcts35, 'b.')
axes[c].plot(np.power(orig_radii50, -alpha), pcts50, 'r.')
axes[c].plot(np.power(orig_radii80, -alpha), pcts80, 'g.')
axes[c].plot(xdata, np.polyval(p35, xdata), color='b')
axes[c].plot(xdata, np.polyval(p50, xdata), color='r')
axes[c].plot(xdata, np.polyval(p80, xdata), color='g')
axes[c].grid(True)
axes[c].set_title("c)", loc="left", fontsize=24)
axes[c].legend([r"$T_{T}$ = 35$^\circ$C",r"$T_{T}$ = 50$^\circ$C", r"$T_{T}$ = 80$^\circ$C"], fontsize=14)
axes[c].set_xlabel('$Radii^{-%.3f}$'%alpha, fontsize=18)
axes[c].set_ylabel(r"Pct(r)", fontsize=18, rotation=90)
axes[c].tick_params(axis='both', which='major', labelsize=16)

#fig.tight_layout()
fig.subplots_adjust(wspace=0.5)
plt.savefig('sse_figure_6.png')


