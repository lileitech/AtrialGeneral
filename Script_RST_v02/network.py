import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F
from layer import U_Encoder2d, U_Decoder_lab2d, CrossTaskmodule2d
from layer import U_Encoder3d, U_Decoder_lab3d, CrossTaskmodule3d
from layer import DoubleConv2d, DoubleConv3d

n_filters = 16
nc = [int(n_filters*(2**i)) for i in range(5)]

class Generator3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Generator3d, self).__init__()

        self.E_feature = U_Encoder3d(in_ch, nc)
        self.D_feature_lab = U_Decoder_lab3d(nc)
        self.D_feature_img = U_Decoder_lab3d(nc)
        self.conv10_seg = nn.Conv3d(nc[0], out_ch, 1)
        self.conv10_img = nn.Conv3d(nc[0], 1, 1)
        self.cross_task = CrossTaskmodule3d(nc)    

    def forward(self, x, y, MIdex_t, MIdex_s):
        input_s = torch.cat([x, MIdex_s-MIdex_t], dim=1)
        F_target = self.E_feature(input_s)
        c9_Tseg = self.D_feature_lab(F_target)
        c9_Timg = self.D_feature_img(F_target)
        c9_Tseg_new = self.cross_task(c9_Tseg, c9_Timg)
        c10_Tseg = self.conv10_seg(c9_Tseg_new)
        out_Tseg = nn.Sigmoid()(c10_Tseg)
        out_Timg = self.conv10_img(c9_Timg)

        input_t = torch.cat([out_Timg, MIdex_t-MIdex_s], dim=1)
        F_source = self.E_feature(input_t)
        c9_Sseg = self.D_feature_lab(F_source)
        c9_Simg = self.D_feature_img(F_source)
        c9_Sseg_new = self.cross_task(c9_Sseg, c9_Simg)
        c10_Sseg = self.conv10_seg(c9_Sseg_new)
        out_Sseg = nn.Sigmoid()(c10_Sseg)
        out_Simg = self.conv10_img(c9_Simg)

        return out_Tseg, out_Timg, out_Sseg, out_Simg

class Discriminator3d(nn.Module):
    def __init__(self, n_modality):
        super(Discriminator3d, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.D_src = nn.Sequential(
            *discriminator_block(1, nc[0], normalize=False),
            *discriminator_block(nc[0], nc[1]),
            *discriminator_block(nc[1], nc[2]),
            *discriminator_block(nc[2], nc[3]),
            #nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv3d(nc[3], 1, 4, padding=1)
        )

        self.D_cls = nn.Sequential(
            *discriminator_block(1, nc[0], normalize=False),
            *discriminator_block(nc[0], nc[1]),
            *discriminator_block(nc[1], nc[2]),
            *discriminator_block(nc[2], nc[3]),
            #nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv3d(nc[3], n_modality, 4, padding=1)
        )

    def forward(self, img):
        out_src = self.D_src(img)
        out_src = F.avg_pool3d(out_src, kernel_size=out_src.size()[2:5])
        out_src = nn.Sigmoid()(out_src)

        out_cls = self.D_cls(img)
        out_cls = F.avg_pool3d(out_cls, kernel_size=out_cls.size()[2:5])
        out_cls = nn.Softmax(dim=1)(out_cls)

        return out_src, out_cls

class Generator2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Generator2d, self).__init__()
        self.E_feature = U_Encoder2d(in_ch, nc)
        self.D_feature_img = U_Decoder_lab2d(nc, out_ch) 

    def forward(self, x, MIdex_t, MIdex_s):
        input_s = torch.cat([x, MIdex_s-MIdex_t], dim=1)
        F_target = self.E_feature(input_s)
        out_Timg = self.D_feature_img(F_target)

        input_t = torch.cat([out_Timg, MIdex_t-MIdex_s], dim=1)
        F_source = self.E_feature(input_t)
        out_Simg = self.D_feature_img(F_source)

        return out_Timg, out_Simg

class Discriminator2d(nn.Module):
    def __init__(self, n_modality):
        super(Discriminator2d, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.D_src = nn.Sequential(
            *discriminator_block(1, nc[0], normalize=False),
            *discriminator_block(nc[0], nc[1]),
            *discriminator_block(nc[1], nc[2]),
            *discriminator_block(nc[2], nc[3]),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(nc[3], 1, 4, padding=1)
        )

        self.D_cls = nn.Sequential(
            *discriminator_block(1, nc[0], normalize=False),
            *discriminator_block(nc[0], nc[1]),
            *discriminator_block(nc[1], nc[2]),
            *discriminator_block(nc[2], nc[3]),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(nc[3], n_modality, 4, padding=1)
        )

    def forward(self, img):
        out_src = self.D_src(img)
        out_src = F.avg_pool2d(out_src, kernel_size=out_src.size()[2:4])
        out_src = nn.Sigmoid()(out_src)

        out_cls = self.D_cls(img)
        out_cls = F.avg_pool2d(out_cls, kernel_size=out_cls.size()[2:4])
        out_cls = nn.Softmax(dim=1)(out_cls)

        return out_src, out_cls

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class Generator2d_new(nn.Module):
    """Generator network."""
    
    def __init__(self, in_ch, out_ch, conv_dim=64, repeat_num=2):
        super(Generator2d, self).__init__()

        layers = []
        layers.append(nn.Conv2d(in_ch, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=3, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, out_ch, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Sigmoid())
        
        self.main = nn.Sequential(*layers)

    def forward(self, x, MIdex_s, MIdex_t, **kwargs):
        # onehot is one-hot vector of domain label with shape (batch, class)
        # c = onehot
        # c = c.view(c.size(0), c.size(1), 1, 1)
        # c = c.repeat(1, 1, x.size(2), x.size(3))
        # x = torch.cat([x, c], dim=1)

        input_s = torch.cat([x, MIdex_s-MIdex_t], dim=1)
        out_Timg = self.main(input_s)

        input_t = torch.cat([out_Timg, MIdex_t-MIdex_s], dim=1)
        out_Simg = self.main(input_t)

        return out_Timg, out_Simg

class U_Net3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(U_Net3d, self).__init__()

        self.conv1 = DoubleConv3d(in_ch, nc[0])
        self.pool1 = nn.MaxPool3d(2, stride=2)
        self.conv2 = DoubleConv3d(nc[0], nc[1])
        self.pool2 = nn.MaxPool3d(2, stride=2)
        self.conv3 = DoubleConv3d(nc[1], nc[2])
        self.pool3 = nn.MaxPool3d(2, stride=2)
        self.conv4 = DoubleConv3d(nc[2], nc[3])
        self.pool4 = nn.MaxPool3d(2, stride=2)
        self.conv5 = DoubleConv3d(nc[3], nc[4])

        self.up6 = nn.ConvTranspose3d(nc[4], nc[3], 2, stride=2)
        self.conv6 = DoubleConv3d(nc[4], nc[3])
        self.up7 = nn.ConvTranspose3d(nc[3], nc[2], 2, stride=2)
        self.conv7 = DoubleConv3d(nc[3], nc[2])
        self.up8 = nn.ConvTranspose3d(nc[2], nc[1], 2, stride=2)
        self.conv8 = DoubleConv3d(nc[2], nc[1])
        self.up9 = nn.ConvTranspose3d(nc[1], nc[0], 2, stride=2)
        self.conv9 = DoubleConv3d(nc[1], nc[0])
        self.conv10 = nn.Conv3d(nc[0], out_ch, 1)

    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)

        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9, c1], dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        out = nn.Sigmoid()(c10)
        #out = nn.Softmax(dim=1)(c10)

        return out 

class U_Net2d(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(U_Net2d, self).__init__()

        self.conv1 = DoubleConv2d(in_ch, nc[0])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv2d(nc[0], nc[1])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv2d(nc[1], nc[2])
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv2d(nc[2], nc[3])
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv2d(nc[3], nc[4])

        self.up6 = nn.ConvTranspose2d(nc[4], nc[3], 2, stride=2)
        self.conv6 = DoubleConv2d(nc[4], nc[3])
        self.up7 = nn.ConvTranspose2d(nc[3], nc[2], 2, stride=2)
        self.conv7 = DoubleConv2d(nc[3], nc[2])
        self.up8 = nn.ConvTranspose2d(nc[2], nc[1], 2, stride=2)
        self.conv8 = DoubleConv2d(nc[2], nc[1])
        self.up9 = nn.ConvTranspose2d(nc[1], nc[0], 2, stride=2)
        self.conv9 = DoubleConv2d(nc[1], nc[0])
        self.conv10 = nn.Conv2d(nc[0], out_ch, 1)

    def forward(self, x):
        #x = torch.cat([x1, x2, x3], dim=1)
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)

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
        c10=self.conv10(c9)
        #out = nn.Softmax(dim=1)(c10)
        out = nn.Sigmoid()(c10)

        return out

