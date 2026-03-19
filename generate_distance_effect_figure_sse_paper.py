
import cv2
import numpy as np
# TODO: figure size, grids, labels, remove axes where it doesn't make sense, colors
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib

'''
Figure is the prediction of practical applications based on the fit: Pct(r) = m*r^-alpha
consider eyes enclosed by near-total face covering allowing a 40mm square, with 5mm pixels on target
40mm at 5mm pixels is 8 pixels diameter, or 4 pixel radius. 
iFOV for the 32deg hFOV boson would be 0.0008726646259971648 radians
and pixel size at distance d is d*np.tan(0.0008726646259971648) = 5.73 meters for pixel size of 5mm
or 1.146 meters for 1mm pixel size
so Pct(r)=0.33*np.power(r, -0.5)

If skin is 35C and ambient is 20C, that becomes:
deviation = 15*Pct(r) = 4.95*np.power(r, -0.5)

# sunny lens iFOV is 25/80
'''

distances = np.linspace(0.5, 6, 100)
# pixel size for Boson with 32deg lens and alpha=0.5, slope=0.33
pixel_size_at_distances = distances * np.tan(0.0008726646259971648)
pixels_to_cover_40mm = 0.04/pixel_size_at_distances
radii_eyeonly = pixels_to_cover_40mm/2.0
deviation_vs_radii = 4.95*np.power(radii_eyeonly, -0.5)
max_distance_cutoff = np.where(pixel_size_at_distances>0.005)[0][0]

# do same for micro80 with Sunny lens and alpha=1, slope=0.15
pixel_size_at_distances_sunny25 = distances * np.tan((25/80)*np.pi/180)
pixels_to_cover_40mm_sunny25  = 0.04/pixel_size_at_distances_sunny25
radii_eyeonly_sunny25  = pixels_to_cover_40mm_sunny25 /2.0
deviation_vs_radii_sunny25  = 15*0.15*np.power(radii_eyeonly_sunny25 , -1)
max_distance_cutoff_sunny25 = np.where(pixel_size_at_distances_sunny25>0.005)[0][0]

# fig 4 subplots a) is graphic of setup (not covered here), b) is deviation vs distance for Boson
# a) is 1/r^alpha with fit, b) is all three with fits, c) is original data with fit inverted for e.g. 35C data
fig = plt.figure(figsize=(8, 6))
axes=fig.gca()
axes.plot(distances, deviation_vs_radii, 'b')
axes.plot(distances, deviation_vs_radii_sunny25, 'r')
ylim=axes.get_ylim()
axes.plot(distances[max_distance_cutoff]*np.ones((100,)), np.linspace(ylim[0], ylim[1], 100), 'b--')
axes.plot(distances[max_distance_cutoff_sunny25]*np.ones((100,)), np.linspace(ylim[0], ylim[1], 100), 'r--')
axes.grid(True)
axes.set_title("b)", loc="left", fontsize=24)
axes.set_xlabel('Distance (m)', fontsize=18)
axes.set_ylabel(r"$\Delta$ $T_{predicted}$ $^\circ$C", fontsize=18, rotation=90)
axes.legend([r'm = 0.33, $\alpha$=0.5, iFOV=0.87mRad', r'm = 0.15, $\alpha$=1, iFOV=5.4mRad', 'Max distance iFOV=0.87mRad', 'Max distance iFOV=5.4mRad'], fontsize=16)
axes.tick_params(axis='both', which='major', labelsize=16)

plt.savefig('sse_figure_Pct_distance.png')



