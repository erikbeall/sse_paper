
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib
dsload = lambda filename: np.load(filename, allow_pickle=True, encoding='latin1').tolist()

# search for alpha providing the best linearity

# load datasets
[orig_radii35, orig_cmaximas35, orig_cbgs35] = np.load('results_stage1_round1_35C.npy')
[orig_radii50, orig_cmaximas50, orig_cbgs50] = np.load('results_stage1_round1_50C.npy')
[orig_radii80, orig_cmaximas80, orig_cbgs80] = np.load('results_stage1_round1_80C.npy')
pcts35 = (np.max(orig_cmaximas35)-orig_cmaximas35)/(np.max(orig_cmaximas35) - orig_cbgs35)
pcts50 = (np.max(orig_cmaximas50)-orig_cmaximas50)/(np.max(orig_cmaximas50) - orig_cbgs50)
pcts80 = (np.max(orig_cmaximas80)-orig_cmaximas80)/(np.max(orig_cmaximas80) - orig_cbgs80)

# linearity -> fit 1d poly, subtract and get residuals
resid35=[]
resid50=[]
resid80=[]
alphas=np.linspace(0.35, 1.5, 1000)
for alpha in alphas:
    p35, residuals, rank, singular_values, rcond=np.polyfit(1/np.power(orig_radii35, alpha), pcts35, 1, full=True)
    resid35.append(residuals[0])
    p50, residuals, rank, singular_values, rcond=np.polyfit(1/np.power(orig_radii50, alpha), pcts50, 1, full=True)
    resid50.append(residuals[0])
    p80, residuals, rank, singular_values, rcond=np.polyfit(1/np.power(orig_radii80, alpha), pcts80, 1, full=True)
    resid80.append(residuals[0])

resid35=np.array(resid35)
resid50=np.array(resid50)
resid80=np.array(resid80)
min_inds=np.where(resid35==np.min(resid35))
min_ind35=int(round(np.mean(min_inds))) if len(min_inds)>1 else min_inds[0]
min_inds=np.where(resid50==np.min(resid50))
min_ind50=int(round(np.mean(min_inds))) if len(min_inds)>1 else min_inds[0]
min_inds=np.where(resid80==np.min(resid80))
min_ind80=int(round(np.mean(min_inds))) if len(min_inds)>1 else min_inds[0]

print('Minimum residuals for alpha=%.2f, %.2f, %.2f'%(alphas[min_ind35], alphas[min_ind50], alphas[min_ind80]))
# p35 gives 0.50
# p50 gives 0.48
# p80 gives 0.42 and different curve

alpha=0.5
# get r-values and fits
p35=np.polyfit(1/np.power(orig_radii35, alpha), pcts35, 1)
r35=np.corrcoef(1/np.power(orig_radii35, alpha), pcts35)[1,0]
p50=np.polyfit(1/np.power(orig_radii50, alpha), pcts50, 1)
r50=np.corrcoef(1/np.power(orig_radii50, alpha), pcts50)[1,0]
p80=np.polyfit(1/np.power(orig_radii80, alpha), pcts80, 1)
r80=np.corrcoef(1/np.power(orig_radii80, alpha), pcts80)[1,0]
xdata=np.linspace(1/np.power(np.min(orig_radii50), alpha), 1/np.power(np.max(orig_radii50), alpha), 100)


# 5 is fits for each with slope and r-value
fig, axes = plt.subplots(1, 3, figsize=(12, 6))
c=0
axes[c].plot(alphas, resid35, 'b')
axes[c].plot(alphas, resid50, 'r')
axes[c].plot(alphas, resid80, 'g')
axes[c].grid(True)
axes[c].set_title("a)", loc="left", fontsize=24)
axes[c].set_xlabel(r'$\alpha$', fontsize=18)
axes[c].set_ylabel(r"Residual", fontsize=18, rotation=90)
axes[c].legend([r"$T_{T}$ = 35$^\circ$C", r"$T_{T}$ = 50$^\circ$C",r"$T_{T}$ = 80$^\circ$C"], fontsize=14)
axes[c].tick_params(axis='both', which='major', labelsize=16)


c=1
alpha=0.5
xdata=np.linspace(1/np.power(np.min(orig_radii80), alpha), 1/np.power(np.max(orig_radii80), alpha), 100)
axes[c].plot(1/np.power(orig_radii80, alpha), pcts80, 'k.')
axes[c].plot(xdata, np.polyval(p80, xdata), color='darkgrey')
axes[c].grid(True)
axes[c].set_title("b)", loc="left", fontsize=24)
axes[c].set_xlabel('$Radii^{-%.02f}$'%alpha, fontsize=18)
axes[c].set_ylabel(r"Pct(r)", fontsize=18, rotation=90)
axes[c].legend([r"$T_{T}$ = 80$^\circ$C", "m=%.2f, r=%.3f"%(p80[0],r80)], fontsize=14)
axes[c].tick_params(axis='both', which='major', labelsize=16)

c=2
# same plot, different alpha
alpha=0.42
xdata=np.linspace(1/np.power(np.min(orig_radii80), alpha), 1/np.power(np.max(orig_radii80), alpha), 100)
p80=np.polyfit(1/np.power(orig_radii80, alpha), pcts80, 1)
r80=np.corrcoef(1/np.power(orig_radii80, alpha), pcts80)[1,0]
axes[c].plot(1/np.power(orig_radii80, alpha), pcts80, 'k.')
axes[c].plot(xdata, np.polyval(p80, xdata), color='darkgrey')
axes[c].grid(True)
axes[c].set_title("c)", loc="left", fontsize=24)
axes[c].set_xlabel('$Radii^{-%.2f}$'%alpha, fontsize=18)
axes[c].set_ylabel(r"Pct(r)", fontsize=18, rotation=90)
axes[c].legend([r"$T_{T}$ = 80$^\circ$C", "m=%.2f, r=%.3f"%(p80[0],r80)], fontsize=14)
axes[c].tick_params(axis='both', which='major', labelsize=16)

#fig.tight_layout()
fig.subplots_adjust(wspace=0.5)
plt.savefig('sse_figure_alphas.png')


