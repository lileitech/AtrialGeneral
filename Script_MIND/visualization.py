import numpy as np
import nibabel as nib
import torch
import os
import pylab as pl


Root_DIR = '/home/lilei/Workspace/AtrialGeneral2021/'
basedir = Root_DIR + 'Script_MIND/lossfile/'

loss = np.loadtxt(basedir + "L_seg.txt") #shapeloss, L_seg
x = range(0, loss.size)
y = loss
pl.subplot(231)
pl.plot(x, y, 'g-', label='L_seg')
pl.legend(frameon=False)

# loss = np.loadtxt(basedir + "L_rec.txt")
# x = range(0, loss.size)
# y = loss
# pl.subplot(232)
# pl.plot(x, y, 'g-', label='L_rec')
# pl.legend(frameon=False)

# loss = np.loadtxt(basedir + "L_dist.txt")
# x = range(0, loss.size)
# y = loss
# pl.subplot(233)
# pl.plot(x, y, 'g-', label='L_dist')
# pl.legend(frameon=False)

pl.show()
pl.savefig('loss_plot.jpg')