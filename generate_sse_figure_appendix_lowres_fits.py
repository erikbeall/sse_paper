
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib
dsload = lambda filename: np.load(filename, allow_pickle=True, encoding='latin1').tolist()

# figure 7 shows fit of each target for one lens, like figure 5 (a,b,c are fits for 35, 50, 80C for one selected lens)
# appendix shows all the rest
lenses=['4mm', 'umi', 'sunny', 'lepton']
lensdata=['39 deg Micro80', '23 deg Micro80', '25 deg Micro80', 'Lepton']
targets=[35, 50, 80]
titles=["a)", "b)", "c)"]
slopes=[0.16, 0.36, 0.16, 0.265]

alpha=1.0
for lens,info in zip(lenses, lensdata):
  fig, axes = plt.subplots(1, 3, figsize=(12, 6))
  for c,setpoint in enumerate(targets):
    [orig_radii, orig_cmaximas, orig_cbgs] = np.load('results_stage1_lens_%s_%dC.npy'%(lens, setpoint))
    pcts = (np.max(orig_cmaximas)-orig_cmaximas)/(np.max(orig_cmaximas) - orig_cbgs)
    p=np.polyfit(1/np.power(orig_radii, alpha), pcts, 1)
    r=np.corrcoef(1/np.power(orig_radii, alpha), pcts)[1,0]
    xdata=np.linspace(1/np.power(np.min(orig_radii), alpha), 1/np.power(np.max(orig_radii), alpha), 100)
    axes[c].plot(np.power(orig_radii, -alpha), pcts, 'k.')
    axes[c].plot(xdata, np.polyval(p, xdata), color='darkgrey')
    axes[c].grid(True)
    axes[c].set_title(titles[c], loc="left", fontsize=24)
    axes[c].set_xlabel(r"$Radii^{-%.1f}$"%alpha, fontsize=18)
    axes[c].set_ylabel(r"Pct(r)", fontsize=18, rotation=90)
    axes[c].legend([r"$T_{T}$ = %d$^\circ$C"%setpoint, "m=%.2f, r=%.3f"%(p[0],r)], fontsize=14)
    axes[c].tick_params(axis='both', which='major', labelsize=16)
  fig.subplots_adjust(wspace=0.5)
  plt.savefig('sse_figure_appendix_fig_%s.png'%info)
  plt.close()

