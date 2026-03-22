
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib
from microbolo_corrections import get_hole_radius_and_edge, get_micro80_ring_background, gen_empirical_kernel
dsload = lambda filename: np.load(filename, allow_pickle=True, encoding='latin1').tolist()


def iterative_correction(nima, kern, iters=4):
    # iters=1 should mean 1 iteration (0-based indexing used in sum below, so add 1)
    iters=iters+1
    c = np.sum(kern)
    bima = cv2.filter2D(nima.astype(np.float32), -1, kern)
    # correction = ima_meas + ima_iter*C - conv(ima_meas,kern)
    # correction_2 = ima_meas + (ima_meas+ima_iter*C - CONV)*C - CONV
    # correction_3 = ima_meas + (ima_meas+(ima_meas+ima_iter*C - CONV)*C - CONV)*C - CONV
    # correction_3 = ima_meas + ima_meas*C + (ima_meas+ima_iter*C - CONV)*C*C - CONV*C - CONV
    # correction_3 = ima_meas + ima_meas*C + ima_meas*C^2 + ima_iter*C^3 - CONV*C^2 - CONV*C - CONV
    # correction_N = ima_meas*sum_0^N[C] - CONV * sum_0^(N-1)[C]
    return np.sum([np.power(c,i) for i in range(iters)])*nima  -  np.sum([np.power(c,i) for i in range(iters-1)])*bima

# same set of data as used in profiles
images=[cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0020.npy')['thermal'], 5)]
images.append(cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0028.npy')['thermal'], 5))
images.append(cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0030.npy')['thermal'], 5))
images.append(cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0033.npy')['thermal'], 5))
images.append(cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0035.npy')['thermal'], 5))
images.append(cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0036.npy')['thermal'], 5))
images.append(cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0037.npy')['thermal'], 5))
images.append(cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0038.npy')['thermal'], 5))

#>>> np.polyfit(1/np.power(orig_radii35, 0.5), pcts35, 1)
#array([0.31750509, 0.00839449])
kern = gen_empirical_kernel(133, 0.5, 0.318)
radii=[]
maxima=[]
cmaxima=[]
for ima in images:
    crop = ima[30:210,120:350]
    num_blurred, dt_slope, max_slope, fit_radius, simple_radius, profile, mmx = get_hole_radius_and_edge(crop, True)
    maxima.append(mmx)
    radii.append(fit_radius)
    cmaxima_iter=[]
    for i in range(0,7):
        cima=iterative_correction(ima, kern, i)
        crop = cima[30:210,120:350]
        num_blurred, dt_slope, max_slope, fit_radius, simple_radius, profile, mmx = get_hole_radius_and_edge(crop, True)
        cmaxima_iter.append(mmx)
    cmaxima.append(cmaxima_iter)


# plot correction versus iteration
cmaxima=np.array(cmaxima)
maxima=np.array(maxima)

# two subplots only, left is uncorrectd, right is corrected
fig, axes = plt.subplots(1, 3, figsize=(12, 6))
c=0
# 1. plot multiple series (a level of iteration vs radii)
axes[c].plot(radii, cmaxima[:,0], 'k')
axes[c].plot(radii, cmaxima[:,1], 'b')
axes[c].plot(radii, cmaxima[:,2], 'g')
axes[c].plot(radii, cmaxima[:,3], 'r')
axes[c].plot(radii, cmaxima[:,4], 'm')
axes[c].plot(radii, cmaxima[:,5], 'k--')
axes[c].legend(["Uncorrected", r"$i$=1",  r"$i$=2",  r"$i$=3",  r"$i$=4",  r"$i$=5"], fontsize=14)
axes[c].set_ylabel(r"$T_{meas}$", fontsize=18, rotation=90)
axes[c].set_xlabel('radii', fontsize=18)
axes[c].grid(True)
axes[c].set_title("a)", loc="left", fontsize=24)
axes[c].tick_params(axis='both', which='major', labelsize=16)

# 2. plot avg and max difference between iterations vs iteration count
c=1
axes[c].plot([np.mean(cmaxima[:,i+1]-cmaxima[:,i]) for i in range(cmaxima.shape[1]-1)], 'darkgrey')
axes[c].plot([np.max(cmaxima[:,i+1]-cmaxima[:,i]) for i in range(cmaxima.shape[1]-1)], 'k')
axes[c].legend([r"$\mu (T^{i+1}_{meas} - T^{i}_{meas})$", r"$max(T^{i+1}_{meas} - T^{i}_{meas})$"], fontsize=14)
axes[c].set_ylabel('successive difference', fontsize=18)
#axes[c].set_xlabel(r"$I^{i+1}_{corrected} - I^{i}_{corrected}$", fontsize=18)
axes[c].set_xlabel(r"$I^{i+1} - I^{i}$", fontsize=18)
axes[c].grid(True)
axes[c].set_title("b)", loc="left", fontsize=24)
axes[c].tick_params(axis='both', which='major', labelsize=16)
axes[c].set_xticks([0,1,2,3,4,5])

# 3. plot ptp and stddev across sizes per iteration (vs iteration count)
c=2
axes[c].plot(np.std(cmaxima,0), 'darkgrey')
axes[c].plot(np.ptp(cmaxima,0), 'k')
axes[c].legend([r"$\sigma (T^{i}_{meas}(r))$", r"$Range (T^{i}_{meas}(r))$"], fontsize=14)
axes[c].set_xlabel(r"iteration count $i$", fontsize=18)
axes[c].set_ylabel('Temperature (C)', fontsize=18)
axes[c].grid(True)
axes[c].set_title("c)", loc="left", fontsize=24)
axes[c].tick_params(axis='both', which='major', labelsize=16)
axes[c].set_xticks([0,1,2,3,4,5,6])



#axes[c].legend([r"$T_{T}$ = 35$^\circ$C"], fontsize=14)
#axes[c].set_xlabel('Horizontal Pixel Position', fontsize=18
#axes[c].set_ylabel(r"$T_{meas}$ $^\circ$C", fontsize=18, rotation=90)

fig.subplots_adjust(wspace=0.35)
plt.savefig('sse_figure_iterations.png')


