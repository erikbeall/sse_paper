
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib
dsload = lambda filename: np.load(filename, allow_pickle=True, encoding='latin1').tolist()

'''
 a) shows the difference between corrected and uncorrected image
 b) shows profile of difference image
larger convolutions are ultimately more effective (e.g. 133 x 133) at avoiding overcorrection with too-high slope needed when the convolution ends early
running search on round1 of each, then applying the found parameters to round2 to assess, then swap direction and compare
convolution must be done in FFT space, for O(X*Y*log(X*Y)) versus O(W*W*X*Y), which is about 3 orders of magnitude faster for 133x133 kernels (only 1-2 orders for 25x25 kernels or approx 30-50 fold)
'''
img7a=cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0020.npy')['thermal'], 5)
img7b=cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0033.npy')['thermal'], 5)
img7c=cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0037.npy')['thermal'], 5)
from microbolo_corrections import gen_empirical_kernel
width=133; alpha=0.675; slope=0.32
kern = gen_empirical_kernel(width, alpha, slope)
def correction(nima, kern, iters=5):
    c = np.sum(kern)
    # raw correction is (ima - conv(ima,kern)) + np.sum(kern)*ima
    # or cima = (ima - conv(ima,kern))
    cima = nima - cv2.filter2D(nima.astype(np.float32), -1, kern)
    # and full correction is cima + np.sum(kern)*ima
    return np.sum([np.power(c,i) for i in range(iters)])*cima + np.power(c, iters)*nima

img7a_c = correction(img7a, kern)
img7b_c = correction(img7b, kern)
img7c_c = correction(img7c, kern)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
c=0
aspect1='equal'
delta_imagea = (img7a_c-img7a)[10:230,120:350]
delta_imageb = (img7b_c-img7b)[10:230,120:350]
delta_imagec = (img7c_c-img7c)[10:230,120:350]
axes[c].imshow(delta_imagea, aspect=aspect1, cmap=matplotlib.colormaps['magma'])
# draw line across and show the profile in the next subplot
axes[c].plot(np.arange(delta_imagea.shape[1]), np.ones(delta_imagea.shape[1])*delta_imagea.shape[0]//2, 'k--')
axes[c].set_title("a)", loc="left", fontsize=24)
axes[c].set_yticklabels([])
axes[c].set_yticks([])
axes[c].set_xticklabels([])
axes[c].set_xticks([])
axes[c].set_xlabel('Crop shown in 1a)', fontsize=18)
#axes[c].axis('off')

c=1
axes[c].plot(delta_imagea[delta_imagea.shape[0]//2, :], 'b')
axes[c].plot(delta_imageb[delta_imagea.shape[0]//2, :], 'r')
axes[c].plot(delta_imagec[delta_imagea.shape[0]//2, :], 'g')
axes[c].legend(["1a)", "1b)", "1c)"], fontsize=14)
axes[c].set_title("b)", loc="left", fontsize=24)
axes[c].grid(True)
axes[c].set_ylabel(r"T$^\circ$C", fontsize=18, rotation=90)
axes[c].set_xlabel('Horizontal Pixel Position', fontsize=18)
axes[c].tick_params(axis='both', which='major', labelsize=16)
#axes[c].set_xlim([0, len(pro1e)])

fig.subplots_adjust(wspace=0.35)
plt.savefig('sse_figure_8.png')

