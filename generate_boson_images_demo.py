

from microbolo_corrections import gen_empirical_kernel, correction
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib

dsload = lambda filename: np.load(filename, allow_pickle=True, encoding='latin1').tolist()

# thermals raw is not calibrated by in-FOV BBs, only by Flir internal routine
ima=(dsload('/home/nano/codebase/thermometry/mar_validations/apr11_acq/faces/image.boson.raw.002.npy')['thermals_raw'][0].astype(np.float32)-27315)/100 + 1.74
kern = gen_empirical_kernel(133, 0.5, 0.305)
cima=correction(np.copy(ima), kern)
# distance was 0.643 meters
distance = dsload('/home/nano/codebase/thermometry/mar_validations/apr11_acq/faces/image.boson.raw.002.npy')['dist']/1000.0
# crop
ima=ima[:300,125:450]
cima=cima[:300,125:450]

# FI data already calibrated with in-FOV blackbodies (and already mSSE-corrected with older mSSE correction - new data must be acquired without either correction)
# distance for micro80 was roughly 1 meter
lowres_cima=np.flipud(np.load('/home/nano/codebase/thermometry/mar_validations/apr11_acq/faces/scan-fi-000049_000309.npy')[-1])
kern = gen_empirical_kernel(55, 1.0, 0.30)
c = np.sum(kern)
fima = cv2.filter2D(lowres_cima.astype(np.float32), -1, kern)
# invert the process
lowres_ima = lowres_cima*(1 - c) + fima
lowres_ima= lowres_ima[:40,15:65]
lowres_cima= lowres_cima[:40,15:65]

# 2 rows, 3 cols is like: fig, axes = plt.subplots(2, 3, figsize=(18, 8))
# options: 2x2 with boson top, micro80 umicore on bottom, or 3x3 with difference image to right
# main image will also have boxes for surface and core estimate and box drawn around eyes
do_3cols=True
if do_3cols:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
else:
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
r=0
c=0
aspect1='equal'
axim=axes[r][c].imshow(ima, aspect=aspect1, cmap=matplotlib.colormaps['magma'], vmin=28, vmax=36)
axes[r][c].set_title("a)", loc="left", fontsize=24)
axes[r][c].set_yticklabels([])
axes[r][c].set_yticks([])
axes[r][c].set_xticklabels([])
axes[r][c].set_xticks([])
axes[r][c].set_xlabel("Uncorrected", fontsize=18)
#axes[r][c].set_xlabel(r"Uncorrected\nSurface=35.3^/circC", fontsize=18)
plt.colorbar(axim, fraction=0.046, pad=0.04)

c=1
axim=axes[r][c].imshow(cima, aspect=aspect1, cmap=matplotlib.colormaps['magma'], vmin=28, vmax=36)
axes[r][c].set_title("b)", loc="left", fontsize=24)
axes[r][c].set_yticklabels([])
axes[r][c].set_yticks([])
axes[r][c].set_xticklabels([])
axes[r][c].set_xticks([])
axes[r][c].set_xlabel('Corrected', fontsize=18)
plt.colorbar(axim, fraction=0.046, pad=0.04)

if do_3cols:
    c=2
    axim=axes[r][c].imshow((cima-ima), aspect=aspect1, cmap=matplotlib.colormaps['magma'])
    axes[r][c].set_title("c)", loc="left", fontsize=24)
    axes[r][c].set_yticklabels([])
    axes[r][c].set_yticks([])
    axes[r][c].set_xticklabels([])
    axes[r][c].set_xticks([])
    axes[r][c].set_xlabel('Difference', fontsize=18)
    plt.colorbar(axim, fraction=0.046, pad=0.04)

r=1
c=0
aspect1='equal'
axim=axes[r][c].imshow(lowres_ima, aspect=aspect1, cmap=matplotlib.colormaps['magma'], vmin=28, vmax=36)
if do_3cols:
    axes[r][c].set_title("d)", loc="left", fontsize=24)
else:
    axes[r][c].set_title("c)", loc="left", fontsize=24)
axes[r][c].set_yticklabels([])
axes[r][c].set_yticks([])
axes[r][c].set_xticklabels([])
axes[r][c].set_xticks([])
axes[r][c].set_xlabel('Uncorrected', fontsize=18)
plt.colorbar(axim, fraction=0.046, pad=0.04)

c=1
axim=axes[r][c].imshow(lowres_cima, aspect=aspect1, cmap=matplotlib.colormaps['magma'], vmin=28, vmax=36)
if do_3cols:
    axes[r][c].set_title("e)", loc="left", fontsize=24)
else:
    axes[r][c].set_title("d)", loc="left", fontsize=24)
axes[r][c].set_yticklabels([])
axes[r][c].set_yticks([])
axes[r][c].set_xticklabels([])
axes[r][c].set_xticks([])
axes[r][c].set_xlabel('Corrected', fontsize=18)
plt.colorbar(axim, fraction=0.046, pad=0.04)

if do_3cols:
    c=2
    axim=axes[r][c].imshow(0.5*(lowres_cima-lowres_ima), aspect=aspect1, cmap=matplotlib.colormaps['magma'])
    axes[r][c].set_title("f)", loc="left", fontsize=24)
    axes[r][c].set_yticklabels([])
    axes[r][c].set_yticks([])
    axes[r][c].set_xticklabels([])
    axes[r][c].set_xticks([])
    axes[r][c].set_xlabel('Difference', fontsize=18)
    plt.colorbar(axim, fraction=0.046, pad=0.04)


fig.subplots_adjust(wspace=0.35)
if do_3cols:
    plt.savefig('sse_figure_demo_3x2.png')
else:
    plt.savefig('sse_figure_demo_2x2.png')

