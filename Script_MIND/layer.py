import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F

import random

class MixStyle(nn.Module):
    """MixStyle.

    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.3, eps=1e-6):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

        print('* MixStyle params')
        print(f'- p: {p}')
        print(f'- alpha: {alpha}')

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        perm = torch.randperm(B)

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix
  
class CrossTaskmodule3d(nn.Module):

    def __init__(self, nc):
        super(CrossTaskmodule3d, self).__init__()
        reduction=8
        self.linear1 = nn.Linear(nc[0], nc[0] // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(nc[0] // reduction, nc[0], bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, F_seg, F_tsl):
        y = F.avg_pool3d(F_tsl, kernel_size=F_tsl.size()[2:5])
        y = y.permute(0, 2, 3, 4, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 4, 1, 2, 3)
        F_tsl_new = F_tsl*y
        F_seg_new = F_seg+F_tsl_new

        return F_seg_new

class U_Encoder3d(nn.Module):
    def __init__(self, in_ch, nc):
        super(U_Encoder3d, self).__init__()

        self.conv1 = DoubleConv3d(in_ch, nc[0])
        self.pool1 = nn.MaxPool3d(2, stride=2)
        self.conv2 = DoubleConv3d(nc[0], nc[1])
        self.pool2 = nn.MaxPool3d(2, stride=2)
        self.conv3 = DoubleConv3d(nc[1], nc[2])
        self.pool3 = nn.MaxPool3d(2, stride=2)
        self.conv4 = DoubleConv3d(nc[2], nc[3])
        self.pool4 = nn.MaxPool3d(2, stride=2)
        self.conv5 = DoubleConv3d(nc[3], nc[4])


    def forward(self, x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        c = c1, c2, c3, c4, c5
        #p = p1, p2, p3, p4

        return c

class U_Decoder_lab3d(nn.Module):
    def __init__(self, nc):
        super(U_Decoder_lab3d, self).__init__()

        self.up6 = nn.ConvTranspose3d(nc[4], nc[3], 2, stride=2)
        self.conv6 = DoubleConv3d(nc[4], nc[3])
        self.up7 = nn.ConvTranspose3d(nc[3], nc[2], 2, stride=2)
        self.conv7 = DoubleConv3d(nc[3], nc[2])
        self.up8 = nn.ConvTranspose3d(nc[2], nc[1], 2, stride=2)
        self.conv8 = DoubleConv3d(nc[2], nc[1])
        self.up9 = nn.ConvTranspose3d(nc[1], nc[0], 2, stride=2)
        self.conv9 = DoubleConv3d(nc[1], nc[0])

    def forward(self, feature):
        c1, c2, c3, c4, c5=feature
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)

        return c9

class U_Decoder_img3d(nn.Module):
    def __init__(self, nc):
        super(U_Decoder_img3d, self).__init__()

        self.up6 = nn.ConvTranspose3d(nc[4], nc[3], 2, stride=2)
        self.conv6 = DoubleConv3d(nc[4], nc[3])
        self.up7 = nn.ConvTranspose3d(nc[3], nc[2], 2, stride=2)
        self.conv7 = DoubleConv3d(nc[3], nc[2])
        self.up8 = nn.ConvTranspose3d(nc[2], nc[1], 2, stride=2)
        self.conv8 = DoubleConv3d(nc[2], nc[1])
        self.up9 = nn.ConvTranspose3d(nc[1], nc[0], 2, stride=2)
        self.conv9 = DoubleConv3d(nc[1], nc[0])

    def forward(self, feature1, feature2):
        c1, c2, c3, c4, c5=feature1
        c5_2=feature2[4]
        c5_12 = c5 + c5_2
        up_6= self.up6(c5_12)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)

        return c9

class CrossTaskmodule2d(nn.Module):

    def __init__(self, nc):
        super(CrossTaskmodule2d, self).__init__()

        reduction=8
        self.linear1 = nn.Linear(nc[0], nc[0] // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(nc[0] // reduction, nc[0], bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, F_seg, F_tsl):
        y = F.avg_pool2d(F_tsl, kernel_size=F_tsl.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        F_tsl_new = F_tsl*y
        F_seg_new = F_seg+F_tsl_new

        return F_seg_new

class U_Encoder2d(nn.Module):
    def __init__(self, in_ch, nc):
        super(U_Encoder2d, self).__init__()

        self.conv1 = DoubleConv2d(in_ch, nc[0])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv2d(nc[0], nc[1])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv2d(nc[1], nc[2])
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv2d(nc[2], nc[3])
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv2d(nc[3], nc[4])


    def forward(self, x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        c = c1, c2, c3, c4, c5
        #p = p1, p2, p3, p4

        return c

class U_Decoder_lab2d(nn.Module):
    def __init__(self, nc):
        super(U_Decoder_lab2d, self).__init__()

        self.up6 = nn.ConvTranspose2d(nc[4], nc[3], 2, stride=2)
        self.conv6 = DoubleConv2d(nc[4], nc[3])
        self.up7 = nn.ConvTranspose2d(nc[3], nc[2], 2, stride=2)
        self.conv7 = DoubleConv2d(nc[3], nc[2])
        self.up8 = nn.ConvTranspose2d(nc[2], nc[1], 2, stride=2)
        self.conv8 = DoubleConv2d(nc[2], nc[1])
        self.up9 = nn.ConvTranspose2d(nc[1], nc[0], 2, stride=2)
        self.conv9 = DoubleConv2d(nc[1], nc[0])

        # self.up6 = nn.ConvTranspose2d(nc[4], nc[3], 2, stride=2)
        # self.conv6 = DoubleConv2d(nc[3], nc[3])
        # self.up7 = nn.ConvTranspose2d(nc[3], nc[2], 2, stride=2)
        # self.conv7 = DoubleConv2d(nc[2], nc[2])
        # self.up8 = nn.ConvTranspose2d(nc[2], nc[1], 2, stride=2)
        # self.conv8 = DoubleConv2d(nc[1], nc[1])
        # self.up9 = nn.ConvTranspose2d(nc[1], nc[0], 2, stride=2)
        # self.conv9 = DoubleConv2d(nc[0], nc[0])

    def forward(self, feature):
        # c5=feature[4]
        # up_6 = self.up6(c5)
        # c6=self.conv6(up_6)
        # up_7=self.up7(c6)
        # c7=self.conv7(up_7)
        # up_8=self.up8(c7)
        # c8=self.conv8(up_8)
        # up_9=self.up9(c8)
        # c9=self.conv9(up_9)

        c1, c2, c3, c4, c5=feature
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9, c1],dim=1)
        c9=self.conv9(merge9)

        return c9

class U_Decoder_img2d(nn.Module):
    def __init__(self, nc):
        super(U_Decoder_img2d, self).__init__()
        self.up6 = nn.ConvTranspose2d(nc[4], nc[3], 2, stride=2)
        self.conv6 = DoubleConv2d(nc[4], nc[3])
        self.up7 = nn.ConvTranspose2d(nc[3], nc[2], 2, stride=2)
        self.conv7 = DoubleConv2d(nc[3], nc[2])
        self.up8 = nn.ConvTranspose2d(nc[2], nc[1], 2, stride=2)
        self.conv8 = DoubleConv2d(nc[2], nc[1])
        self.up9 = nn.ConvTranspose2d(nc[1], nc[0], 2, stride=2)
        self.conv9 = DoubleConv2d(nc[1], nc[0])

    def forward(self, feature1, feature2):
        c1, c2, c3, c4, c5=feature1
        c5_2=feature2[4]
        c5_12 = c5 + c5_2
        up_6= self.up6(c5_12)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)

        return c9

class DoubleConv_Leaky3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv_Leaky3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class DoubleConv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class DoubleConv_Leaky2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv_Leaky2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class DoubleConv2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)
