
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib
dsload = lambda filename: np.load(filename, allow_pickle=True, encoding='latin1').tolist()

# lepton, 4mm, umicore, sunny
# figure 7 shows fit of each target for one lens, like figure 5 (a,b,c are fits for 35, 50, 80C for one selected lens)
# appendix shows all the rest
lenses=['4mm', 'umi', 'sunny', 'lepton']
lensdata=['39 deg Micro80', '23 deg Micro80', '25 deg Micro80', 'Lepton']
targets=[35, 50, 80]

# get a quick estimate on alpha with log-log fitting
for lens,info in zip(lenses, lensdata):
  plt.clf()
  legends=[]
  for setpoint in targets:
    [orig_radii, orig_cmaximas, orig_cbgs] = np.load('results_stage1_lens_%s_%dC.npy'%(lens, setpoint))
    tgt=np.max(orig_cmaximas)
    pcts = (tgt-orig_cmaximas)/(tgt - np.mean(orig_cbgs))
    xdata=np.log(orig_radii)
    ydata=np.log(pcts)
    ydata = np.log(pcts) #1 + 100*((tgt-orig_cmaximas)/(tgt-orig_cbgs)))
    inds=xdata<2.25
    p = np.polyfit(xdata[inds], ydata[inds],1)
    legends.append(r'$T_T$=%d$^\circ$C $\alpha$=%.2f'%(setpoint, p[0]))
    plt.ylabel('log(Pct+eps)')
    plt.xlabel('log(radius)')
    plt.title('Lens %s'%(info))
    plt.plot(xdata[inds], ydata[inds],'.')
  plt.legend(legends)
  plt.savefig('loglog_%s.png'%lens)

import os
os._exit(0)
plt.close()
[orig_radii35, orig_cmaximas35, orig_cbgs35] = np.load('results_stage1_lens_umi_35C.npy')
[orig_radii50, orig_cmaximas50, orig_cbgs50] = np.load('results_stage1_lens_umi_50C.npy')
[orig_radii80, orig_cmaximas80, orig_cbgs80] = np.load('results_stage1_lens_umi_80C.npy')
orig_radii35=orig_radii35[25:]; orig_cmaximas35=orig_cmaximas35[25:]; orig_cbgs35=orig_cbgs35[25:]
pcts35 = (np.max(orig_cmaximas35)-orig_cmaximas35)/(np.max(orig_cmaximas35) - orig_cbgs35)
pcts50 = (np.max(orig_cmaximas50)-orig_cmaximas50)/(np.max(orig_cmaximas50) - orig_cbgs50)
pcts80 = (np.max(orig_cmaximas80)-orig_cmaximas80)/(np.max(orig_cmaximas80) - orig_cbgs80)
# get r-values and fits
alpha=1.0
p35=np.polyfit(1/np.power(orig_radii35, alpha), pcts35, 1)
r35=np.corrcoef(1/np.power(orig_radii35, alpha), pcts35)[1,0]
p50=np.polyfit(1/np.power(orig_radii50, alpha), pcts50, 1)
r50=np.corrcoef(1/np.power(orig_radii50, alpha), pcts50)[1,0]
p80=np.polyfit(1/np.power(orig_radii80, alpha), pcts80, 1)
r80=np.corrcoef(1/np.power(orig_radii80, alpha), pcts80)[1,0]
xdata=np.linspace(1/np.power(np.min(orig_radii50), alpha), 1/np.power(np.max(orig_radii50), alpha), 100)

# 7 is fits for each with slope and r-value
fig, axes = plt.subplots(1, 3, figsize=(12, 6))
c=0
axes[c].plot(np.power(orig_radii35, -alpha), pcts35, 'k.')
axes[c].plot(xdata, np.polyval(p35, xdata), color='darkgrey')
axes[c].grid(True)
axes[c].set_title("a)", loc="left", fontsize=24)
axes[c].set_xlabel(r"$Radii^{-%.1f}$"%alpha, fontsize=18)
axes[c].set_ylabel(r"Pct(r)", fontsize=18, rotation=90)
axes[c].legend([r"$T_{T}$ = 35$^\circ$C", "m=%.2f, r=%.3f"%(p35[0],r35)], fontsize=14)
axes[c].tick_params(axis='both', which='major', labelsize=16)

c=1
axes[c].plot(np.power(orig_radii50, -alpha), pcts50, 'k.')
axes[c].plot(xdata, np.polyval(p50, xdata), color='darkgrey')
axes[c].grid(True)
axes[c].set_title("b)", loc="left", fontsize=24)
axes[c].set_xlabel('$Radii^{-1.0}$', fontsize=18)
axes[c].set_ylabel(r"Pct(r)", fontsize=18, rotation=90)
print(axes[c].get_xlim())
print(axes[c].get_ylim())
axes[c].legend([r"$T_{T}$ = 50$^\circ$C", "m=%.2f, r=%.3f"%(p50[0],r50)], fontsize=14)
axes[c].tick_params(axis='both', which='major', labelsize=16)

c=2
axes[c].plot(np.power(orig_radii80, -alpha), pcts80, 'k.')
axes[c].plot(xdata, np.polyval(p80, xdata), color='darkgrey')
#axes[c].set_ylim([-0.00825, 0.175])
#axes[c].set_xlim([-1.5, 90])
axes[c].grid(True)
axes[c].set_title("c)", loc="left", fontsize=24)
axes[c].set_xlabel('$Radii^{-1.0}$', fontsize=18)
axes[c].set_ylabel(r"Pct(r)", fontsize=18, rotation=90)
axes[c].legend([r"$T_{T}$ = 80$^\circ$C", "m=%.2f, r=%.3f"%(p80[0],r80)], fontsize=14)
axes[c].tick_params(axis='both', which='major', labelsize=16)

#fig.tight_layout()
fig.subplots_adjust(wspace=0.5)
plt.savefig('sse_figure_7.png')


