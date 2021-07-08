import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F
from layer import U_Encoder2d, U_Decoder_lab2d, U_Decoder_img2d
from layer import U_Encoder3d, U_Decoder_lab3d, U_Decoder_img3d
from layer import MixStyle
import segmentation_models_pytorch as smp

n_filters = 16
nc = [int(n_filters*(2**i)) for i in range(5)]


# DeepLabV3Plus2d_seg = smp.DeepLabV3Plus('efficientnet-b6', in_channels=1, classes=1, activation='sigmoid')
# DeepLabV3Plus2d_img = smp.DeepLabV3Plus('efficientnet-b6', in_channels=1, classes=1)

def list_add(a,b):
    c = []
    for i in range(len(a)):
        c.append(a[i]+b[i])
    return c         

class MIDNet3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MIDNet3d, self).__init__()

        self.E_feature1 = U_Encoder3d(in_ch, nc)
        self.E_feature2 = U_Encoder3d(in_ch, nc)        
        self.D_feature_lab = U_Decoder_lab3d(nc)
        self.D_feature_img = U_Decoder_img3d(nc)
        self.conv10_seg = nn.Conv3d(nc[0], out_ch, 1)
        self.conv10_img = nn.Conv3d(nc[0], 1, 1)  

    def forward(self, x):
        
        Feature_lab = self.E_feature1(x)
        Feature_img = self.E_feature2(x)
        c9_seg = self.D_feature_lab(Feature_lab)
        c9_img = self.D_feature_img(Feature_img, Feature_lab)
        c10_seg = self.conv10_seg(c9_seg)
        out_seg = nn.Sigmoid()(c10_seg)
        out_img = self.conv10_img(c9_img)

        return out_seg, out_img, Feature_lab, Feature_img

class MIDNet2d_new(nn.Module):
    def __init__(self,             
            encoder_name: str = "efficientnet-b6",
            in_channels: int = 3,
            classes: int = 1,):
        super(MIDNet2d, self).__init__()

        self.DeepLabV3Plus2d_seg = smp.DeepLabV3Plus(encoder_name=encoder_name, in_channels=in_channels, classes=classes, activation='sigmoid') 
        self.E_feature1 = self.DeepLabV3Plus2d_seg.encoder 
        self.D_feature_lab = self.DeepLabV3Plus2d_seg.decoder    
        self.conv_seg = self.DeepLabV3Plus2d_seg.segmentation_head       
        self.mixstyle = MixStyle(p=0.5, alpha=0.1)

    def forward(self, x):       
        #Feature_lab = self.E_feature1(x)
        stages = self.E_feature1.get_stages() 
        block_number = 0.
        drop_connect_rate = self.E_feature1._global_params.drop_connect_rate

        Feature_lab = []
        for i in range(self.E_feature1._depth + 1):
            # Identity and Sequential stages
            if i < 2:
                x = stages[i](x)
                #x = self.mixstyle(x)
            # Block stages need drop_connect rate
            else:
                for module in stages[i]:
                    drop_connect = drop_connect_rate * block_number / len(self.E_feature1._blocks)
                    block_number += 1.
                    x = module(x, drop_connect)
                    #x = self.mixstyle(x)
            Feature_lab.append(x)

        decoder_seg = self.D_feature_lab(*Feature_lab)
        out_seg = self.conv_seg(decoder_seg)

        return out_seg

class MIDNet2d(nn.Module):
    def __init__(self,             
            encoder_name: str = "efficientnet-b6",
            in_channels: int = 3,
            classes: int = 1,):
        super(MIDNet2d, self).__init__()

        self.DeepLabV3Plus2d_seg = smp.DeepLabV3Plus(encoder_name=encoder_name, in_channels=in_channels, classes=classes, activation='sigmoid')
        self.DeepLabV3Plus2d_img = smp.DeepLabV3Plus(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
        
        self.E_feature1 = self.DeepLabV3Plus2d_seg.encoder
        self.E_feature2 = self.DeepLabV3Plus2d_img.encoder       
        self.D_feature_lab = self.DeepLabV3Plus2d_seg.decoder
        self.D_feature_img = self.DeepLabV3Plus2d_img.decoder        
        self.conv_seg = self.DeepLabV3Plus2d_seg.segmentation_head    
        self.conv_img = self.DeepLabV3Plus2d_img.segmentation_head 

    def forward(self, x):
        
        Feature_lab = self.E_feature1(x)
        Feature_img = self.E_feature2(x)
        Feature_img_lab = list_add(Feature_lab, Feature_img)
        decoder_seg = self.D_feature_lab(*Feature_lab)
        decoder_img = self.D_feature_img(*Feature_img_lab)
        out_seg = self.conv_seg(decoder_seg)
        out_img = self.conv_img(decoder_img)

        return out_seg, out_img, Feature_lab, Feature_img

class U_Net2d(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Seg_2DNet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64,out_ch, 1)

        # self.SA = SAmodule(1024)
        # self.CA = CAmodule(1024)

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)

        # c5_SA = self.SA(c5)
        # c5_CA = self.CA(c5)

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
        out = nn.Softmax(dim=1)(c10)

        return out
