
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib
dsload = lambda filename: np.load(filename, allow_pickle=True, encoding='latin1').tolist()


lenses=['4mm', 'umi', 'sunny', 'lepton']
lensdata=['39 deg Micro80', '23 deg Micro80', '25 deg Micro80', 'Lepton']
targets=[35, 50, 80]

for lens,info in zip(lenses, lensdata):
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    [radii35, maximas35, cbgs35, maximas35_c, bgs35_c] = np.load('results_stage2_lens_%s_%dC.npy'%(lens, 35))
    [radii50, maximas50, cbgs50, maximas50_c, bgs50_c] = np.load('results_stage2_lens_%s_%dC.npy'%(lens, 50))
    [radii80, maximas80, cbgs80, maximas80_c, bgs80_c] = np.load('results_stage2_lens_%s_%dC.npy'%(lens, 80))
    pcts35 = (np.max(maximas35)-maximas35)/(np.max(maximas35) - cbgs35)
    pcts50 = (np.max(maximas50)-maximas50)/(np.max(maximas50) - cbgs50)
    pcts80 = (np.max(maximas80)-maximas80)/(np.max(maximas80) - cbgs80)
    pcts35_c = (np.max(maximas35_c)-maximas35_c)/(np.max(maximas35_c) - bgs35_c)
    pcts50_c = (np.max(maximas50_c)-maximas50_c)/(np.max(maximas50_c) - bgs50_c)
    pcts80_c = (np.max(maximas80_c)-maximas80_c)/(np.max(maximas80_c) - bgs80_c)
    tgt35 = np.polyfit(1/np.power(radii35[radii35>2.5], 1), maximas35[radii35>2.5], 1)[1]
    tgt50 = np.polyfit(1/np.power(radii50[radii50>2.5], 1), maximas50[radii50>2.5], 1)[1]
    tgt80 = np.polyfit(1/np.power(radii80[radii80>2.5], 1), maximas80[radii80>2.5], 1)[1]
    tgt35_c = np.polyfit(1/np.power(radii35[radii35>2.5], 1), maximas35_c[radii35>2.5], 1)[1]
    tgt50_c = np.polyfit(1/np.power(radii50[radii50>2.5], 1), maximas50_c[radii50>2.5], 1)[1]
    tgt80_c = np.polyfit(1/np.power(radii80[radii80>2.5], 1), maximas80_c[radii80>2.5], 1)[1]
    pcts35 = (tgt35-maximas35)/(tgt35 - cbgs35)
    pcts50 = (tgt50-maximas50)/(tgt50 - cbgs50)
    pcts80 = (tgt80-maximas80)/(tgt80 - cbgs80)
    pcts35_c = (tgt35_c-maximas35_c)/(tgt35_c - bgs35_c)
    pcts50_c = (tgt50_c-maximas50_c)/(tgt50_c - bgs50_c)
    pcts80_c = (tgt80_c-maximas80_c)/(tgt80_c - bgs80_c)

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
    #axes[c].plot(np.power(radii50, -1), pcts50, marker='.', color='k', linewidth=0)
    #axes[c].plot(np.power(radii50, -1), pcts50_c, marker='.', color='darkgrey', linewidth=0)
    axes[c].plot(radii50, pcts50, marker='.', color='k', linewidth=0)
    axes[c].plot(radii50, pcts50_c, marker='.', color='darkgrey', linewidth=0)
    axes[c].grid(True)
    axes[c].set_title("b)", loc="left", fontsize=24)
    axes[c].set_xlabel('Radii', fontsize=18)
    axes[c].set_ylabel(r"Pct(r)", fontsize=18, rotation=90)
    axes[c].legend([r"$T_{meas}$", r"$T_{corr}$"], fontsize=12)
    axes[c].tick_params(axis='both', which='major', labelsize=16)
    
    c=2
    axes[c].plot(radii35, pcts35, marker='.', color='k', linewidth=0)
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
    axes[c].legend(["Uncorrected (3)", r"$T_{T}$ = 35$^\circ$C",r"$T_{T}$ = 50$^\circ$C", r"$T_{T}$ = 80$^\circ$C"], fontsize=14)
    axes[c].set_title("c)", loc="left", fontsize=24)
    axes[c].set_xlabel('Radii', fontsize=18)
    axes[c].set_ylabel(r"Pct(r)", fontsize=18, rotation=90)
    axes[c].tick_params(axis='both', which='major', labelsize=16)
    fig.subplots_adjust(wspace=0.5)
    plt.savefig('sse_figure_appendix_lowres_results_%s.png'%info)
    plt.close()



