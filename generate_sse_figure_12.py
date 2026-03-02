
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib
from microbolo_corrections import gen_empirical_kernel

kern_m80 = gen_empirical_kernel(55, 1.0, 0.3)
kern_boson = gen_empirical_kernel(133, 0.5, 0.3)

kern2=kern_boson[kern_boson.shape[0]//2,:]
kern1=kern_m80[kern_m80.shape[0]//2,:]


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
c=0
aspect1='equal'
w=kern_boson.shape[0]//2
dw=5
axes[c].imshow(kern_boson[w-dw:w+dw+1, w-dw:w+dw+1], aspect=aspect1, cmap=matplotlib.colormaps['turbo'])
axes[c].set_title("a)", loc="left", fontsize=24)
axes[c].set_yticklabels([])
axes[c].set_yticks([])
axes[c].set_xticklabels([])
axes[c].set_xticks([])
axes[c].plot(np.arange(dw*2+1), np.ones(dw*2+1)*dw, 'r--')
axes[c].set_xlabel('Central %dx%d'%(2*dw,2*dw), fontsize=18)

c=1

axes[c].plot(kern2[len(kern2)//2-5:len(kern2)//2+6])
axes[c].plot(kern1[len(kern1)//2-5:len(kern1)//2+6])
axes[c].legend([r"$\alpha$=0.5, m=0.3", r"$\alpha$=1.0, m=0.3"], fontsize=14)
axes[c].set_title("b)", loc="left", fontsize=24)
axes[c].grid(True)
axes[c].set_ylabel(r"$w_{i,j}", fontsize=18, rotation=90)
axes[c].set_xlabel('Central 10 indices', fontsize=18)
axes[c].set_xticks(np.arange(11))
axes[c].set_xticklabels(['-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5'])
axes[c].tick_params(axis='both', which='major', labelsize=16)

fig.subplots_adjust(wspace=0.35)
plt.savefig('sse_figure_12.png')




