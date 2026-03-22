
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
ima = dsload('image.boson.R002.0018.npy')['thermal']
kern = gen_empirical_kernel(133, 0.5, 0.305)
cima=correction(np.copy(ima), kern)
# distance was 1.1 meters (note, boson extends 30mm past the TOF plane)
distance = dsload('image.boson.R002.0018.npy')['dist'] - 0.03
# crop
ima=ima[20:280,170:430]
cima=cima[20:280,170:430]

oral = 36.4
ambient=22.0

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
c=0
aspect1='equal'
axim=axes[c].imshow(ima, aspect=aspect1, cmap=matplotlib.colormaps['magma'], vmin=28, vmax=36)
axes[c].set_title("a)", loc="left", fontsize=24)
axes[c].set_yticklabels([])
axes[c].set_yticks([])
axes[c].set_xticklabels([])
axes[c].set_xticks([])
axes[c].set_xlabel("Uncorrected", fontsize=18)
#axes[c].set_xlabel(r"Uncorrected\nSurface=35.3^/circC", fontsize=18)
plt.colorbar(axim, fraction=0.046, pad=0.04)
# eyes_rect=ptch.Rectangle((70, 100), 110, 50, linewidth=1, edgecolor='b', facecolor='none')
# axes[c].add_patch(eyes_rect)
# eyes_image = ima[100:100+50, 70:70+110]
lrect=ptch.Rectangle((107, 113), 8, 15, linewidth=1, edgecolor='b', facecolor='none')
axes[c].add_patch(lrect)
rrect=ptch.Rectangle((133, 113), 8, 15, linewidth=1, edgecolor='r', facecolor='none');
axes[c].add_patch(rrect)
lcanthi=np.median(np.sort(ima[113:113+15, 107:107+8].flatten())[-8:])
rcanthi=np.median(np.sort(ima[113:113+15, 133:133+8].flatten())[-8:])
lcore = (lcanthi - ambient)*0.1 + lcanthi
rcore = (rcanthi - ambient)*0.1 + rcanthi
# add pixel probe
textstr = '\n'.join([r'$T_{LCanthi}=%.2f$'%lcanthi, r'$T_{EstCore}=%.2f$'%lcore, r'$T_{OralCore}=%.2f$'%oral])
props=dict(boxstyle='round', facecolor='white', alpha=0.5)
axes[c].text(0.05, 0.95, textstr, transform=axes[c].transAxes, fontsize=14, verticalalignment='top', bbox=props)
textstr = '\n'.join([r'$T_{RCanthi}=%.2f$'%rcanthi, r'$T_{EstCore}=%.2f$'%rcore, r'$T_{OralCore}=%.2f$'%oral])
props=dict(boxstyle='round', facecolor='white', alpha=0.5)
axes[c].text(0.55, 0.95, textstr, transform=axes[c].transAxes, fontsize=14, verticalalignment='top', bbox=props)

c=1
axim=axes[c].imshow(cima, aspect=aspect1, cmap=matplotlib.colormaps['magma'], vmin=28, vmax=36)
axes[c].set_title("b)", loc="left", fontsize=24)
axes[c].set_yticklabels([])
axes[c].set_yticks([])
axes[c].set_xticklabels([])
axes[c].set_xticks([])
axes[c].set_xlabel('Corrected', fontsize=18)
plt.colorbar(axim, fraction=0.046, pad=0.04)
lrect=ptch.Rectangle((107, 113), 8, 15, linewidth=1, edgecolor='b', facecolor='none')
axes[c].add_patch(lrect)
rrect=ptch.Rectangle((133, 113), 8, 15, linewidth=1, edgecolor='r', facecolor='none');
axes[c].add_patch(rrect)
lcanthi=np.median(np.sort(cima[113:113+15, 107:107+8].flatten())[-8:])
rcanthi=np.median(np.sort(cima[113:113+15, 133:133+8].flatten())[-8:])
lcore = (lcanthi - ambient)*0.1 + lcanthi
rcore = (rcanthi - ambient)*0.1 + rcanthi
# add pixel probe
textstr = '\n'.join([r'$T_{LCanthi}=%.2f$'%lcanthi, r'$T_{EstCore}=%.2f$'%lcore, r'$T_{OralCore}=%.2f$'%oral])
props=dict(boxstyle='round', facecolor='white', alpha=0.5)
axes[c].text(0.05, 0.95, textstr, transform=axes[c].transAxes, fontsize=14, verticalalignment='top', bbox=props)
textstr = '\n'.join([r'$T_{RCanthi}=%.2f$'%rcanthi, r'$T_{EstCore}=%.2f$'%rcore, r'$T_{OralCore}=%.2f$'%oral])
props=dict(boxstyle='round', facecolor='white', alpha=0.5)
axes[c].text(0.55, 0.95, textstr, transform=axes[c].transAxes, fontsize=14, verticalalignment='top', bbox=props)

c=2
axim=axes[c].imshow((cima-ima), aspect=aspect1, cmap=matplotlib.colormaps['magma'])
axes[c].set_title("c)", loc="left", fontsize=24)
axes[c].set_yticklabels([])
axes[c].set_yticks([])
axes[c].set_xticklabels([])
axes[c].set_xticks([])
axes[c].set_xlabel('Difference', fontsize=18)
plt.colorbar(axim, fraction=0.046, pad=0.04)
lrect=ptch.Rectangle((107, 113), 8, 15, linewidth=1, edgecolor='b', facecolor='none')
axes[c].add_patch(lrect)
rrect=ptch.Rectangle((133, 113), 8, 15, linewidth=1, edgecolor='r', facecolor='none');
axes[c].add_patch(rrect)
lcanthi=np.mean(np.abs((cima-ima)[113:113+15, 107:107+8].flatten()))
rcanthi=np.mean(np.abs((cima-ima)[113:113+15, 133:133+8].flatten()))
# add pixel probe
textstr = '\n'.join([r'$MAD_{LCanthi}=%.2f$'%lcanthi, r'$MAD_{RCanthi}=%.2f$'%rcanthi])
props=dict(boxstyle='round', facecolor='white', alpha=0.5)
axes[c].text(0.35, 0.95, textstr, transform=axes[c].transAxes, fontsize=14, verticalalignment='top', bbox=props)

fig.subplots_adjust(wspace=0.35)
plt.savefig('sse_figure_visualimpact_redux.png')

