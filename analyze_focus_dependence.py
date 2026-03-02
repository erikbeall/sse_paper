
# session effects (reference blackbody process value, movement of ROIs, shutter)
# equilibration effects (vtemp, delta vtemp over time)
# verify each blackbody value with NCIT
# record distance to target for each configuration

def get_hole_radius_and_edge(ncrop):
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
    mmx =  np.mean(ncrop[crops[-1]==1])
    pcts = np.array([(mmx-np.mean(ncrop[(crops[i] - crops[i+1])==1]))/np.ptp(ncrop) for i in range(len(crops)-1)])
    # where the curve turns up is where diffraction takes over from partial pixel effects (and on boson this is probably a focus effect
    dpct = pcts[1:]-pcts[:-1]
    # simple metric, how many pixels are below half the ptp of delta pct
    num_blurred = np.sum(dpct<0.5*(np.max(dpct)+np.min(dpct)))
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
        print('failed to find contour')
        #raise Exception
        fit_radius=simple_radius
        profiles=[[0],[0]]
    return num_blurred, dt_slope, max_slope, fit_radius, simple_radius, profiles, mmx

allmaximas=[]
allpcts=[]
allradii=[]
dirs=['round2', 'round3', 'round4_blurfirstlevel', 'round5_blursecondlevel']
names=['subideal focus (w=6)', 'ideal focus', 'subideal focus (w=3)', 'subideal focus (w=5)']
for i in range(len(dirs)):
    d=dirs[i]
    relinearize=True
    if d=='round2':
        relinearize=False
    files=glob.glob(d+'/image*npy')
    files.sort()
    #crop=np.copy(dsload(files[0])['thermal'][0:220,:])
    # 20:230, 140:340
    crop=np.copy(dsload(files[0])['thermal'][20:230, 140:340])
    ambient=dsload(files[0])['amb']
    crop[crop<(np.max(crop)-2)]=np.min(crop)
    crop=crop-np.min(crop)
    crop[crop>0] = 1
    contours,_ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    center, axes, smth = cv2.fitEllipse(contours[0])
    # empirically set bounds
    #x1,x2,y1,y2 = [round(center[1]-80),round(center[1]+80),round(center[0]-80),round(center[0]+80)]
    x1,x2,y1,y2 = [round(center[1]-100),round(center[1]+100),round(center[0]-100),round(center[0]+100)]
    # find center
    radii=[]
    maxima=[]
    profiles=[]
    blurreds=[]
    for f in files:
        crop=dsload(f)['thermal'][x1:x2,y1:y2]
        if relinearize:
            imabuf=dsload(f)['thermal']
            # images contain pair of TEC-controlled blackbodies
            bbc_set=31.25
            bbc_set=24
            bbh_set=36.25
            # hardcoded setpoints and locations
            bbc=np.median(np.sort(imabuf[270:410,10:80].flatten())[125:-50])
            bbh=np.median(np.sort(imabuf[300:400,480:570].flatten())[125:-50])
            bb_mid=0.5*(bbc+bbh)
            bb_set_mid=0.5*(bbc_set+bbh_set)
            scaler = (bbh_set - bbc_set) / (bbh - bbc)
            imabuf = ((imabuf - bb_mid) * scaler) + bb_set_mid
            crop= ((crop- bb_mid) * scaler) + bb_set_mid
        try:
            num_blurred, dt_slope, max_slope, fit_radius, simple_radius, profile, mmx = get_hole_radius_and_edge(crop)
            blurreds.append(num_blurred)
            radii.append(fit_radius)
            maxima.append(mmx)
            profiles.append(profile)
        except:
            print('skipping ', f)
    maxima=np.array(maxima)
    radii=np.array(radii)
    maxima=maxima[(radii>1)*(radii<100)]
    radii=radii[(radii>1)*(radii<100)]
    mx=np.max(maxima)+0.25
    pcts = (mx-maxima)/(mx-ambient)
    allmaximas.append(maxima)
    allradii.append(radii)
    allpcts.append(pcts)
    plt.figure()
    plt.plot(radii, pcts, '.')
    plt.ylim([0, 0.15])
    plt.title('Percent Reduction with %s'%names[i])

# the radii are all well above the pixel-diffractive boundary (e.g. > 60 pixels within 10% of maxima)
# and down to a radius of 3.5 pixels there is minimal difference even with 
# noticeably bad focus blur (I do expect it to get worse however, but only at the radii where focus blur takes over)


