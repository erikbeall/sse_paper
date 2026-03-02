
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib
from microbolo_corrections import get_hole_radius_and_edge, get_micro80_ring_background, gen_empirical_kernel, correction
dsload = lambda filename: np.load(filename, allow_pickle=True, encoding='latin1').tolist()

aspect1='equal'
images=[cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0020.npy')['thermal'], 5)]
images.append(cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0028.npy')['thermal'], 5))
images.append(cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0030.npy')['thermal'], 5))
images.append(cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0033.npy')['thermal'], 5))
images.append(cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0035.npy')['thermal'], 5))
images.append(cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0036.npy')['thermal'], 5))
images.append(cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0037.npy')['thermal'], 5))
images.append(cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0038.npy')['thermal'], 5))

kern = gen_empirical_kernel(133, 0.5, 0.305)
radii=[]
maxima=[]
cmaxima=[]
for ima in images:
    crop = ima[30:210,120:350]
    num_blurred, dt_slope, max_slope, fit_radius, simple_radius, profile, mmx = get_hole_radius_and_edge(crop, True)
    maxima.append(mmx)
    cima=correction(ima, kern)
    crop = cima[30:210,120:350]
    num_blurred, dt_slope, max_slope, fit_radius, simple_radius, profile, mmx = get_hole_radius_and_edge(crop, True)
    radii.append(fit_radius)
    cmaxima.append(mmx)

# two subplots only, left is uncorrectd, right is corrected
fig, axes = plt.subplots(1, 2, figsize=(8, 6))
c=0
# 11a is uncorrected profiles
for ima in images:
    axes[c].plot(ima[122,140:340])

#axes[c].legend([r"$T_{T}$ = 35$^\circ$C"], fontsize=14)
axes[c].grid(True)
axes[c].set_title("a)", loc="left", fontsize=24)
axes[c].set_xlabel('Horizontal Pixel Position', fontsize=18)
axes[c].set_ylabel(r"$T_{meas}$ $^\circ$C", fontsize=18, rotation=90)
axes[c].tick_params(axis='both', which='major', labelsize=16)


c=1
for ima in images:
    axes[c].plot(correction(ima, kern)[122,140:340])

#axes[c].legend([r"$T_{T}$ = 35$^\circ$C"], fontsize=14)
axes[c].grid(True)
axes[c].set_title("b)", loc="left", fontsize=24)
axes[c].set_xlabel('Horizontal Pixel Position', fontsize=18)
axes[c].set_ylabel(r"$T_{corrected}$ $^\circ$C", fontsize=18, rotation=90)
axes[c].tick_params(axis='both', which='major', labelsize=16)

fig.subplots_adjust(wspace=0.35)
plt.savefig('sse_figure_11.png')


