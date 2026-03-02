
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib
dsload = lambda filename: np.load(filename, allow_pickle=True, encoding='latin1').tolist()

aspect1='equal'
img1a=cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0020.npy')['thermal'], 5)[30:210,120:350]
img1b=cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0033.npy')['thermal'], 5)[30:210,120:350]
img1c=cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0037.npy')['thermal'], 5)[30:210,120:350]
# 1a is large disk
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
mn=np.min(img1a); mx=np.max(img1a)
c=0; r=0;
axes[r][c].imshow(img1a, aspect=aspect1, cmap=matplotlib.colormaps['magma'], vmin = np.min(img1a), vmax=np.max(img1a))
axes[r][c].plot(np.arange(img1a.shape[1]), np.ones(img1a.shape[1])*img1a.shape[0]//2, 'r--')
axes[r][c].set_title("a)", loc="left", fontsize=24)
axes[r][c].axis('off')
#axes[0][0].axis('tight')
print('maxima: ', mx)

# 1d is mask 1a) and erode per methods
data=dsload('new_fixture_35C/round1/image.boson.R002.0020.npy')
ncrop = cv2.medianBlur(data['thermal'][30:210,120:350], 5)
crop=np.copy(ncrop) 
crop[crop<np.max(crop)-0.5*np.ptp(crop)]=np.min(crop)
crop[crop>np.min(crop)]=np.max(crop)
crop = crop-np.min(crop)
crop[crop>0]=1
simple_radius = np.sqrt((np.sum(crop)/np.pi))
# shrink this by pixel until within 0.3C or percent
crops=[cv2.dilate(crop, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3+(i*2),3+(i*2))), iterations=1) for i in range(5)]
crops.reverse()
crops.append(crop)
crops.extend([cv2.erode(crop, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3+(i*2),3+(i*2))), iterations=1) for i in range(5)])
crops = [c for c in crops if np.sum(c)>8]
# this maxima will be sensitive to smaller radii since the fraction of partial-covered pixels will dominate below radii of 3
# fraction of edge vs total = 2pi*r/(pi*r^2) = 2/r, which crosses below 50% at r>=4
# alternative method is to exclude analyses below pixel radii of 3
vals = np.sort(ncrop[crops[-1]==1])
img1d=crops[-1]
mmx =  np.median(vals[len(vals)//2:])
c=0; r=1;
axes[r][c].imshow(img1d, aspect=aspect1, cmap=matplotlib.colormaps['gray'])
axes[r][c].set_title("d)", loc="left", fontsize=24)
axes[r][c].axis('off')
print('crop 1d): ', mmx)

# 1b is mid size
mn=np.min(img1a); mx=np.max(img1a)
c=1; r=0;
axes[r][c].imshow(img1b, aspect=aspect1, cmap=matplotlib.colormaps['magma'], vmin = np.min(img1b), vmax=np.max(img1b))
axes[r][c].plot(np.arange(img1b.shape[1]), np.ones(img1b.shape[1])*img1b.shape[0]//2, 'r--')
axes[r][c].set_title("b)", loc="left", fontsize=24)
axes[r][c].axis('off')
#axes[1][0].axis('tight')
print('maxima 1b): ', np.max(img1b))

# 1c is small size
mn=np.min(img1c); mx=np.max(img1c)
c=2; r=0;
axes[r][c].imshow(img1c, aspect=aspect1, cmap=matplotlib.colormaps['magma'], vmin = np.min(img1c), vmax=np.max(img1c))
axes[r][c].plot(np.arange(img1c.shape[1]), np.ones(img1c.shape[1])*img1c.shape[0]//2, 'r--')
axes[r][c].set_title("c)", loc="left", fontsize=24)
axes[r][c].axis('off')
#axes[2][0].axis('tight')
print('maxima 1c): ', np.max(img1c))

# 1e is three profiles
#pro1e=cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0020.npy')['thermal'], 5)[122, 85:400]
pro1e=cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0020.npy')['thermal'], 5)[122, 140:340]
pro2e=cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0033.npy')['thermal'], 5)[122, 140:340]
pro3e=cv2.medianBlur(dsload('new_fixture_35C/round1/image.boson.R002.0037.npy')['thermal'], 5)[122, 140:340]
c=1; r=1;
axes[r][c].plot(pro1e, 'b')
axes[r][c].plot(pro2e, 'r')
axes[r][c].plot(pro3e, 'g')
axes[r][c].legend(["1a)", "1b)", "1c)"], fontsize=14)
axes[r][c].set_title("e)", loc="left", fontsize=24)
axes[r][c].grid(True)
#axes[r][c].inset_axes()
axes[r][c].set_ylabel(r"T$^\circ$C", fontsize=18, rotation=90)
axes[r][c].set_xlabel('Horizontal Pixel Position', fontsize=18)
axes[r][c].tick_params(axis='both', which='major', labelsize=16)
axes[r][c].set_xlim([0, len(pro1e)])
#axes[r][c].set_aspect('equal')
#axes[r][c].box_aspect(1.0)

# 1f is collected maxima of full set of 35C boson data vs radius
[orig_radii, orig_cmaximas, orig_cbgs] = np.load('results_stage1_round1_35C.npy')
# first 25 datapoints contained fixture movement and a one-off datapoint outlier
c=2; r=1
axes[r][c].plot(orig_radii[25:], orig_cmaximas[25:], 'k.')
axes[r][c].legend([r"$T_T$ = 35$^\circ$C"], fontsize=14)
axes[r][c].grid(True)
axes[r][c].set_xlim([-5, 90])
axes[r][c].set_title("f)", loc="left", fontsize=24)
axes[r][c].set_xlabel('Radii', fontsize=18)
axes[r][c].set_ylabel(r"$T_{meas}$ $^\circ$C", fontsize=18, rotation=90)
axes[r][c].tick_params(axis='both', which='major', labelsize=16)
axes[r][c].set_ylim([32.75, 34.75])

#fig.tight_layout()
fig.subplots_adjust(wspace=0.35)
plt.savefig('sse_figure_1.png')


