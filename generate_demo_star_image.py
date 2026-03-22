
#22.25 ambient, 46.9% RH by the DHT20 sensor
#97.6F core by WelchAllyn
#22.0C ambient, 47.5% RH by the SHT41 sensor

from microbolo_corrections import gen_empirical_kernel, correction
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from matplotlib import colormaps
import matplotlib

dsload = lambda filename: np.load(filename, allow_pickle=True, encoding='latin1').tolist()

# image used was pre-calibrated by dual-ETRS system
ima = dsload('image.boson.R003.0015.npy')['thermal']
ima_full = dsload('image.boson.R003.0035.npy')['thermal']
kern = gen_empirical_kernel(133, 0.5, 0.305)
cima=correction(np.copy(ima), kern)
cima_full=correction(np.copy(ima_full), kern)
# crop
ima=ima[60:220, 220:380]
cima=cima[60:220, 220:380]
ima_full=ima_full[70:230, 220:380]
cima_full=cima_full[70:230, 220:380]

# median filter for masking
mima=cv2.medianBlur(ima, 3)
mima_full=cv2.medianBlur(ima_full, 3)
mcima=cv2.medianBlur(cima, 3)
mcima_full=cv2.medianBlur(cima_full, 3)

# decent profile of a spoke, use median filter with minimal width (shape showing the deviation is unaltered)
profile_spoke_uncorrected = mima[78,20:80]
profile_spoke_corrected = mcima[78,20:80]
profile_fullima = mcima_full[78,20:80]

# a) shows spoke image, b) shows corrected with inset rect on both, c) shows profiles

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
c=0
aspect1='equal'
axim=axes[c].imshow(ima, aspect=aspect1, cmap=matplotlib.colormaps['magma'], vmin=32, vmax=36.5)
axes[c].set_title("a)", loc="left", fontsize=24)
axes[c].set_yticklabels([])
axes[c].set_yticks([])
axes[c].set_xticklabels([])
axes[c].set_xticks([])
axes[c].set_xlabel("Uncorrected", fontsize=18)
plt.colorbar(axim, fraction=0.046, pad=0.04)
axes[c].plot(np.arange(20,80), np.ones(len(profile_spoke_corrected))*78, 'r--')
# add pixel probe
#textstr = '\n'.join([r'$T_{RCanthi}=%.2f$'%rcanthi, r'$T_{EstCore}=%.2f$'%rcore, r'$T_{OralCore}=%.2f$'%oral])
#props=dict(boxstyle='round', facecolor='white', alpha=0.5)
#axes[c].text(0.55, 0.95, textstr, transform=axes[c].transAxes, fontsize=14, verticalalignment='top', bbox=props)

c=1
axim=axes[c].imshow(cima, aspect=aspect1, cmap=matplotlib.colormaps['magma'], vmin=32, vmax=36.5)
axes[c].set_title("b)", loc="left", fontsize=24)
axes[c].set_yticklabels([])
axes[c].set_yticks([])
axes[c].set_xticklabels([])
axes[c].set_xticks([])
axes[c].set_xlabel('Corrected', fontsize=18)
plt.colorbar(axim, fraction=0.046, pad=0.04)
axes[c].plot(np.arange(20,80), np.ones(len(profile_spoke_corrected))*78, 'r--')

c=2
axes[c].plot(profile_fullima, 'k--')
axes[c].plot(profile_spoke_uncorrected, 'k')
axes[c].plot(profile_spoke_corrected, 'darkgrey')
axes[c].set_title("c)", loc="left", fontsize=24)

axes[c].grid(True)
axes[c].set_ylabel(r"Temperature", fontsize=18, rotation=90)

axes[c].set_xlabel('Pixel position', fontsize=18)
axes[c].legend(['Full-field', 'Uncorrected', 'Corrected'])

fig.subplots_adjust(wspace=0.35)
plt.savefig('sse_figure_visualimpact_redux_star.png')

