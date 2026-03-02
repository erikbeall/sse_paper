
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib
dsload = lambda filename: np.load(filename, allow_pickle=True, encoding='latin1').tolist()


# uncorrected and corrected data for two sets of data each
[radii35, maximas35, bgs35, maximas35_c, bgs35_c] = np.load('results_stage2_round1_35C.npy')
[radii50, maximas50, bgs50, maximas50_c, bgs50_c] = np.load('results_stage2_round1_50C.npy')
[radii80, maximas80, bgs80, maximas80_c, bgs80_c] = np.load('results_stage2_round1_80C.npy')
[radii35_round2, maximas35_round2, bgs35_round2, maximas35_c_round2, bgs35_c_round2] = np.load('results_stage2_round2_35C.npy')
[radii50_round2, maximas50_round2, bgs50_round2, maximas50_c_round2, bgs50_c_round2] = np.load('results_stage2_round2_50C.npy')
radii50_round2=radii50_round2[20:]; maximas50_round2=maximas50_round2[20:]; bgs50_round2=bgs50_round2[20:]
#[radii80_round2, maximas80_round2, bgs80_round2, maximas80_c_round2, bgs80_c_round2] = np.load('results_stage2_round2_80C.npy')
# pcts - use target from corrected data fit
tgt35_c = np.polyfit(1/np.power(radii35[radii35>2.5], 0.5), maximas35_c[radii35>2.5], 1)[1]
tgt50_c = np.polyfit(1/np.power(radii50[radii50>2.5], 0.5), maximas50_c[radii50>2.5], 1)[1]
tgt80_c = np.polyfit(1/np.power(radii80[radii80>2.5], 0.5), maximas80_c[radii80>2.5], 1)[1]
pcts35 = (tgt35_c-maximas35)/(tgt35_c - bgs35)
pcts50 = (tgt50_c-maximas50)/(tgt50_c - bgs50)
pcts80 = (tgt80_c-maximas80)/(tgt80_c - bgs80)
pcts35_c = (tgt35_c-maximas35_c)/(tgt35_c - bgs35_c)
pcts50_c = (tgt50_c-maximas50_c)/(tgt50_c - bgs50_c)
pcts80_c = (tgt80_c-maximas80_c)/(tgt80_c - bgs80_c)


'''
Figure 9 is the corrected data
 a) shows maxima(r) and maxima_corrected(r) for selected data
 b) shows Pct(r) and Pct_corrected(r) for same
 c) shows Pct_corrected(r) for all three (perhaps offset)
'''
# for images (and images mixed with plots), ensure equal aspect ratio (e.g. 12,6 for 1,2 subplots - figure uses c,r, not r,c)
# for plots only, use 4,6 per subplot, so 12,6 for 1,3 subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 6))

c=0
axes[c].plot(radii50, maximas50, marker='.', color='k', linewidth=0)
axes[c].plot(radii50, maximas50_c, marker='.', color='darkgrey', linewidth=0)
axes[c].grid(True)
axes[c].set_title("a)", loc="left", fontsize=24)
axes[c].set_xlabel('Radii', fontsize=18)
axes[c].set_ylabel(r"$T_{meas}$ $^\circ$C", fontsize=18, rotation=90)
axes[c].legend([r"$T_{meas}$", r"$T_{corr}$"], fontsize=12)
axes[c].tick_params(axis='both', which='major', labelsize=16)

c=1
axes[c].plot(radii50, pcts50, marker='.', color='k', linewidth=0)
axes[c].plot(radii50, pcts50_c, marker='.', color='darkgrey', linewidth=0)
axes[c].grid(True)
axes[c].set_title("b)", loc="left", fontsize=24)
axes[c].set_xlabel('Radii', fontsize=18)
axes[c].set_ylabel(r"Pct(r)", fontsize=18, rotation=90)
axes[c].legend([r"$T_{meas}$", r"$T_{corr}$"], fontsize=12)
axes[c].tick_params(axis='both', which='major', labelsize=16)


s35=np.std(maximas35_c)
s50=np.std(maximas50_c)
s80=np.std(maximas80_c)
c=2
s35=np.std(pcts35_c)
s50=np.std(pcts50_c)
s80=np.std(pcts80_c)
axes[c].plot(radii35, pcts35_c, marker='.', color='b', linewidth=0)
axes[c].plot(radii50, pcts50_c, marker='.', color='r', linewidth=0)
axes[c].plot(radii80, pcts80_c, marker='.', color='g', linewidth=0)
axes[c].plot(radii35, pcts35, marker='.', color='k', linewidth=0)
axes[c].plot(radii50, pcts50, marker='.', color='k', linewidth=0)
axes[c].plot(radii80, pcts80, marker='.', color='k', linewidth=0)
axes[c].plot(radii35, pcts35_c, marker='.', color='b', linewidth=0)
axes[c].plot(radii50, pcts50_c, marker='.', color='r', linewidth=0)
axes[c].plot(radii80, pcts80_c, marker='.', color='g', linewidth=0)
axes[c].grid(True)
axes[c].legend([r"$T_{T}$ = 35$^\circ$C",r"$T_{T}$ = 50$^\circ$C", r"$T_{T}$ = 80$^\circ$C"], fontsize=14)
axes[c].set_title("c)", loc="left", fontsize=24)
axes[c].set_xlabel('Radii', fontsize=18)
axes[c].set_ylabel(r"$T_{meas}$ $^\circ$C", fontsize=18, rotation=90)
axes[c].tick_params(axis='both', which='major', labelsize=16)

#fig.tight_layout()
fig.subplots_adjust(wspace=0.5)
plt.savefig('sse_figure_9.png')


