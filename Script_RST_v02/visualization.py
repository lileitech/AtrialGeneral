import numpy as np
import nibabel as nib
import torch
import os
import pylab as pl


Root_DIR = '/home/lilei/Workspace/AtrialGeneral2021/'
basedir = Root_DIR + 'Script_RST_v02/lossfile/'

net = 'Seg'
# net = 'GAN'
if net=='GAN':
    loss = np.loadtxt(basedir + "L_shape.txt") #shapeloss, L_seg
    x = range(0, loss.size)
    y = loss
    pl.subplot(241)
    pl.plot(x, y, 'g-', label='L_shape')
    pl.legend(frameon=False)

    loss = np.loadtxt(basedir + "L_rec.txt")
    x = range(0, loss.size)
    y = loss
    pl.subplot(242)
    pl.plot(x, y, 'g-', label='L_rec')
    pl.legend(frameon=False)

    loss = np.loadtxt(basedir + "L_novel.txt")
    x = range(0, loss.size)
    y = loss
    pl.subplot(243)
    pl.plot(x, y, 'g-', label='L_novel')
    pl.legend(frameon=False)

    loss = np.loadtxt(basedir + "L_adv_G.txt")
    x = range(0, loss.size)
    y = loss
    pl.subplot(244)
    pl.plot(x, y, 'g-', label='L_adv_G')
    pl.legend(frameon=False)

    loss = np.loadtxt(basedir + "L_adv_D.txt")
    x = range(0, loss.size)
    y = loss
    pl.subplot(245)
    pl.plot(x, y, 'r-', label='L_adv_D')
    pl.legend(frameon=False)

    loss = np.loadtxt(basedir + "L_cls_r.txt")
    x = range(0, loss.size)
    y = loss
    pl.subplot(246)
    pl.plot(x, y, 'g-', label='L_cls_r')
    pl.legend(frameon=False)

    loss = np.loadtxt(basedir + "L_cls_f.txt")
    x = range(0, loss.size)
    y = loss
    pl.subplot(247)
    pl.plot(x, y, 'r-', label='L_cls_f')
    pl.legend(frameon=False)
else:
    loss = np.loadtxt(basedir + "L_shape.txt") #shapeloss, L_seg
    x = range(0, loss.size)
    y = loss
    pl.subplot(241)
    pl.plot(x, y, 'g-', label='total loss')
    pl.legend(frameon=False)

    loss = np.loadtxt(basedir + "L_rec.txt")
    x = range(0, loss.size)
    y = loss
    pl.subplot(242)
    pl.plot(x, y, 'g-', label='loss_f')
    pl.legend(frameon=False)

    loss = np.loadtxt(basedir + "L_novel.txt")
    x = range(0, loss.size)
    y = loss
    pl.subplot(243)
    pl.plot(x, y, 'g-', label='loss_r')
    pl.legend(frameon=False)

    loss = np.loadtxt(basedir + "L_adv_G.txt")
    x = range(0, loss.size)
    y = loss
    pl.subplot(244)
    pl.plot(x, y, 'g-', label='loss_predicted r')
    pl.legend(frameon=False)


pl.show()
pl.savefig('loss_plot.jpg')