
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib
dsload = lambda filename: np.load(filename, allow_pickle=True, encoding='latin1').tolist()

# get a quick estimate on alpha with log-log fitting
plt.clf()
legends=[]
targets=[35, 50, 80]
for setpoint in targets:
    [orig_radii, orig_cmaximas, orig_cbgs] = np.load('results_stage1_round1_%dC.npy'%setpoint)
    if setpoint==35:
        # discard first 55 datapoints of round1 35C setpoint data
        orig_radii=orig_radii[55:]; orig_cmaximas=orig_cmaximas[55:]; orig_cbgs=orig_cbgs[55:]
    elif setpoint==50:
        orig_radii=orig_radii[20:]; orig_cmaximas=orig_cmaximas[20:]; orig_cbgs=orig_cbgs[20:]
    # however, we do not at this point know the alpha to use - we aim to obtain it here
    # nevertheless, to perform a meaningful log-log plot, we need a best estimate of the maxima
    #tgt=np.polyfit(1/np.power(orig_radii, would_require_an_assumed_alpha), orig_cmaximas, 1)[1]
    # however, this drives the fitted alpha to the value used here
    # therefore, use the maximum
    tgt=np.max(orig_cmaximas)
    pcts = (tgt - orig_cmaximas)/(tgt - np.mean(orig_cbgs))
    xdata=np.log(orig_radii)
    # fit to 1 plus percentage scaled to tractable (comparable) range
    ydata=np.log(1 + 30*pcts)
    inds=xdata<3.25
    p = np.polyfit(xdata[inds], ydata[inds],1)
    legends.append(r'$T_T$=%d$^\circ$C $\alpha$=%.2f'%(setpoint, p[0]))
    plt.ylabel('log(Pct+eps)')
    plt.xlabel('log(radius)')
    plt.title('Boson')
    #plt.plot(xdata[inds], ydata[inds],'.')
    plt.plot(xdata, ydata,'.')
    print('slope = %.3f'%np.polyfit(1/np.power(orig_radii, 0.5), pcts, 1)[0])
plt.legend(legends)
plt.savefig('loglog_boson.png')

