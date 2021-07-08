import numpy as np
import nibabel as nib
import torch
import os
import pylab as pl

Root_DIR = '/home/lilei/Workspace/AtrialGeneral2021/'

lossfile_DIR = Root_DIR + 'Script_net/lossfile/'

lossfile = lossfile_DIR + '/loss.txt'
lossfile1 = lossfile_DIR + '/loss_1.txt'
lossfile2 = lossfile_DIR + '/loss_2.txt'


loss = np.loadtxt(lossfile)
x = range(0, loss.size)
y = loss
pl.subplot(231)
pl.plot(x, y, 'g-', label='total loss')
pl.legend(frameon=False)


loss = np.loadtxt(lossfile1)
x = range(0, loss.size)
y = loss
pl.subplot(232)
pl.plot(x, y, 'g-', label='LA BCE loss')
pl.legend(frameon=False)


loss = np.loadtxt(lossfile2)
x = range(0, loss.size)
y = loss
pl.subplot(233)
pl.plot(x, y, 'g-', label='LA Dice loss')
pl.legend(frameon=False)


pl.show()
pl.savefig(Root_DIR + "/test.jpg")