
import numpy as np
import cv2

def gen_empirical_kernel(width=9, alpha=1, slope=0.25): #r_offset=1):
    # width must be odd
    assert (width-1)//2 == int(round((width-1)/2.0)), 'Must use odd-valued width'
    r = np.arange(width)
    # apply circular mask so the kernel is circular
    a,b = ((width-1)/2.0, (width-1)/2.0)
    y,x = np.ogrid[-a:width-a,-b:width-b]
    mask = (x*x + y*y) <= a*b+2 #1.5
    kernel2d = np.zeros((width, width))
    # 0.159 factor is 1/2pi
    weight_r = lambda radius, alpha: 0.1591549 * alpha * np.power(radius, -float(alpha)-2)
    for x in range(width):
        for y in range(width):
            radius = np.sqrt((a-x)**2 + (b-y)**2)
            # apply a weight for every pixel except central
            weight = slope*weight_r(radius, alpha) if radius>0 else 0
            if np.isinf(weight) or np.isnan(weight) or radius>a+1:
                weight=0.0
            kernel2d[x,y] = weight
    # no normalization - the alpha and fitted slope with the weight equation are all that are needed
    #kernel2d=kernel2d/np.sum(kernel2d)
    return kernel2d

def get_truncated_kernel(kernel, width):
    r=(width-1)//2
    c=kernel.shape[0]//2
    return np.copy(kernel)[max(0,c-r):c+r+1, max(0,c-r):c+r+1]

def get_hole_radius_and_edge(ncrop, return_vals_when_fail=False):
    # median blur to reduce noise
    ncrop=cv2.medianBlur(ncrop, 5)
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
    #mmx =  np.mean(ncrop[crops[-1]==1])
    # alternative method is to exclude analyses below pixel radii of 3
    vals = np.sort(ncrop[crops[-1]==1])
    mmx =  np.median(vals[len(vals)//2:])
    pcts = np.array([(mmx-np.mean(ncrop[(crops[i] - crops[i+1])==1]))/np.ptp(ncrop) for i in range(len(crops)-1)])
    # where the curve turns up is where diffraction takes over from partial pixel effects (and on boson this is probably a focus effect
    dpct = pcts[1:]-pcts[:-1]
    # simple metric, how many pixels are below half the ptp of delta pct
    num_blurred = np.sum(dpct<0.5*(np.max(dpct)+np.min(dpct)))
    # dpct is the change from one pixel to the next as we change radius (by repeated erode/dilate operations)
    # so the sharper the image focus is, the larger and more negative the max_slope will be (for perfect focus this could be -1)
    # while dt_slope will increase in magnitude as well
    # dt_slope is the difference between max slope and the average of the adjacent slopes, which for perfect focus those will be 0
    mean_slope = np.mean(dpct[(dpct<0.5*(np.max(dpct)+np.min(dpct)))])
    max_slope = np.min(dpct)
    ind=np.where(dpct==max_slope)[0][0]
    try:
        dt_slope = 0.5*(dpct[ind+1]+dpct[ind-1]) - max_slope
    except:
        dt_slope=0
    # 10% ends up giving me the exact same results as above (albeit without a measurement of the focus
    crop=np.copy(ncrop)
    crop[crop<np.max(crop)-0.1*np.ptp(crop)]=np.min(crop)
    crop[crop>np.min(crop)]=np.max(crop)
    crop = crop-np.min(crop)
    crop[crop>0]=1
    simple_radius = np.sqrt((np.sum(crop)/np.pi))
    simple_center=np.mean(np.where(crop==1),1)
    try:
        contours,_ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        center, axes, smth = cv2.fitEllipse(contours[0])
        delta=round(center[1]-center[0])
        profiles=[ncrop[round(center[1]),:], ncrop[:, round(center[0])]]
        if delta<0:
            profiles[0] = profiles[0][abs(delta):]
            profiles[1] = profiles[1][:-abs(delta)]
        elif delta>0:
            profiles[1] = profiles[1][abs(delta):]
            profiles[0] = profiles[0][:-abs(delta)]
        if len(profiles[0])>len(profiles[1]):
            profiles[0] = profiles[0][:len(profiles[1])]
        elif len(profiles[0])<len(profiles[1]):
            profiles[1] = profiles[1][:len(profiles[0])]
        profile = np.mean(profiles,0)
        fit_radius = 0.5*np.mean(axes)
    except:
        if return_vals_when_fail:
            fit_radius=simple_radius
            profiles=[[0],[0]]
        else:
            raise Exception
    return num_blurred, dt_slope, max_slope, fit_radius, simple_radius, profiles, mmx

def get_micro80_ring_background(ncrop):
    # get ring around central region
    ncrop=cv2.medianBlur(np.copy(ncrop), 5)
    crop=np.copy(ncrop)
    crop[crop<np.max(crop)-0.25*np.ptp(crop)]=np.min(crop)
    crop[crop>np.min(crop)]=np.max(crop)
    crop = crop-np.min(crop)
    crop[crop>0]=1
    # expand this by 9x9 - these are for moderate-to-high resolution systems only (e.g. not for 80x60 or 80x80)
    crop = cv2.dilate(crop, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    outercrop = cv2.dilate(crop, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
    background = np.median(ncrop[(outercrop-crop)==1])
    return background

# use scipy-based (full convolution)
def correction_sci(nima, kern, iters=5):
    c = np.sum(kern)
    bima = scipysignal.convolve2d(nima.astype(np.float32), kern, mode='same', boundary='symm')
    return np.sum([np.power(c,i) for i in range(iters)])*nima  -  np.sum([np.power(c,i) for i in range(iters-1)])*bima

# use opencv, which switches between full and FFT-based depending on size (here will _always_ use FFT due to the size of our kernels)
def correction(nima, kern, iters=5):
    c = np.sum(kern)
    bima = cv2.filter2D(nima.astype(np.float32), -1, kern)
    return np.sum([np.power(c,i) for i in range(iters)])*nima  -  np.sum([np.power(c,i) for i in range(iters-1)])*bima

# this can run into convergence issues if alpha is 0.5 or less, not recommended to use this as-is
def get_smallest_kernel_needed(kernel, delta_t=15, required_accuracy=0.1, size_of_source=3):
    kern=np.copy(kernel)
    # center
    c=kern.shape[0]//2
    offsets=[i for i in range(1,c)]
    shapes=[i*2+1 for i in offsets]
    # near worst case artifact, with source having pixel radius of 3
    # how big does the kernel have to be outside of this to make a big difference?
    radii_image=1+0*np.copy(kern)
    cf=(kern.shape[0]-1)/2
    for x in range(width):
        for y in range(width):
            xr=cf-x
            yr=cf-y
            if np.sqrt(xr**2 + yr**2) < size_of_source:
                radii_image[x,y]=0
            else:
                radii_image[x,y]=1
    kern_midzeroed = np.copy(kern)*radii_image
    # fraction of weights producing a fractional effect on data
    correction_impact = np.array([delta_t*np.sum(kern_midzeroed[max(0,c-r):c+r+1, max(0,c-r):c+r+1]) for r in offsets])
    residual_error_vs_radii = correction_impact[-1] - correction_impact
    sizes=[kern[max(0,c-r):c+r+1, max(0,c-r):c+r+1].shape[0] for r in offsets]
    required_size = sizes[np.where(residual_error_vs_radii<required_accuracy)[0][0]]
    r = radii[np.where(residual_error_vs_radii<required_accuracy)[0][0]]
    return kernel[max(0,c-r):c+r+1, max(0,c-r):c+r+1]

# 1/r^alpha is not separable enough, this is not recommended unless you intentionally perturb the kernel, as various perturbations may be more separable and still get good results
# be mindful of the delta between separable applied in two different directions (row then col, vs col then row) - if you see shifts, those are artifacts introduced by using a non-separable kernel
def get_separable_kernel(kern):
    from separate_kernel import separate_kernel
    res=separate_kernel(kern, symmetric_kernel=True, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    reco_kern=res.col_vec @ res.row_vec
    print(np.sum(np.abs((reco_kern-kern).flatten())))
    print(np.sum((reco_kern-kern).flatten()))
    print(np.std((reco_kern-kern).flatten()))
    return reco_kern

