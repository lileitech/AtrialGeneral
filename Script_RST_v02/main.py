import numpy as np
import torch
from torch import nn
import nibabel as nib
from torch.utils.data import DataLoader
import torch.utils.data as data
from torch import optim
import os
import time
import glob
from scipy import stats
from torch.nn import DataParallel
from torch.backends import cudnn
import collections
import sys
import segmentation_models_pytorch as smp

from loaddata import LoadDataset2d, LoadDataset3d, ProcessTestDataset2d
from network import U_Net2d, Generator3d, Discriminator3d, Generator2d, Discriminator2d
from function import F_LoadParam_test, F_LoadParam, F_loss_G, F_loss_D, F_loss, F_gradient_penalty, F_mkdir

datasetname = '*_1' #IC, Utah_I, KCL, Yale, *_1
post_pre = '*' #Post, Pre
Root_DIR = '/home/lilei/Workspace/AtrialGeneral2021/'

Data_DIR = Root_DIR + 'Data/' + datasetname + '/' + post_pre + '/'
TRAIN_DIR_PATH = Data_DIR + 'train_data/'
#TEST_DIR_PATH = Data_DIR + 'test_data/'
TEST_DIR_PATH = Root_DIR + 'Data/Utah_I/' + post_pre + '/test_data/'
TRAIN_SAVE_DIR_Seg = Root_DIR + 'Script_RST_v02/result_model/'
TRAIN_SAVE_DIR_best = Root_DIR + 'Script_RST_v02/best_model/'


lossfile_DIR = Root_DIR + 'Script_RST_v02/lossfile/'
lossfile = lossfile_DIR + 'L_shape.txt'
lossfile1 = lossfile_DIR + 'L_rec.txt'
lossfile2 = lossfile_DIR + 'L_novel.txt'
lossfile3 = lossfile_DIR + 'L_adv_G.txt'
lossfile4 = lossfile_DIR + 'L_adv_D.txt'
lossfile5 = lossfile_DIR + 'L_cls_r.txt'
lossfile6 = lossfile_DIR + 'L_cls_f.txt'

WORKERSNUM = 8 #16
BatchSize = 4 #16, 10, 7
NumEPOCH = 100
LEARNING_RATE = 5e-5#1e-3
REGULAR_RATE = 0.95
WEIGHT_DECAY = 1e-4
n_modality = 3

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print('Lei is using ' + str(device))
testonecase = False

class TrainingDataset(data.Dataset):
    def __init__(self, datapath):
        self.datafile = glob.glob(datapath + '*')
        self.numpyimage = []
        self.numpylabel = []
        self.numpyMIndex = []
        self.NumOfSubjects = 0

        for subjectid in range(len(self.datafile)):
            if testonecase == True:
                if subjectid > 1:
                    break
            imagename = self.datafile[subjectid] + '/enhanced.nii.gz'
            labelname = self.datafile[subjectid] + '/atriumSegImgMO.nii.gz'

            print('loading training image: ' + imagename)
            numpyimage, numpylabel, NumSlice = LoadDataset2d(imagename, labelname)
            
            if imagename.find('IC_1') != -1:
                numpyMIndex = np.array([1, 0, 0])
            elif imagename.find('KCL_1') != -1:
                numpyMIndex = np.array([0, 1, 0])
            elif imagename.find('Yale_1') != -1:
                numpyMIndex = np.array([0, 0, 1])
            numpyMIndex_new = [[numpyMIndex] for i in range(NumSlice)] 
                         
            self.numpyMIndex.extend(numpyMIndex_new)
            self.numpyimage.extend(numpyimage)
            self.numpylabel.extend(numpylabel)
            self.NumOfSubjects += NumSlice 
            
    def __getitem__(self, item):

        numpyimage = np.array([self.numpyimage[item]])
        numpylabel = np.array([self.numpylabel[item]])
        numpylabel = (numpylabel > 0) * 1

        tensorimage = torch.from_numpy(numpyimage).float()
        tensorlabel = torch.from_numpy(numpylabel.astype(np.float32))

        numpyMIndex_source = np.array(self.numpyMIndex[item])
        tensorMIndex_s = torch.from_numpy(numpyMIndex_source).float().squeeze()

        numpyMIndex_target = np.eye(n_modality, dtype=int)[np.random.randint(0, n_modality-1)] #random code for target domain
        tensorMIndex_t = torch.from_numpy(np.array(numpyMIndex_target)).float()

        return tensorimage, tensorlabel, tensorMIndex_s, tensorMIndex_t

    def __len__(self):
        return self.NumOfSubjects

def Train_Validate_Seg(dataload, G_net, S_net, epoch, S_optimizer, savedir):
    start_time = time.time()
    flearning_rate = LEARNING_RATE*(REGULAR_RATE**(epoch//50))
    fregular_rate = 1.0
    f = open(lossfile, 'a')
    f1 = open(lossfile1, 'a')
    f2 = open(lossfile2, 'a')
    f3 = open(lossfile3, 'a')

    d_step = list(range(0,10)) + list(range(20,30)) + list(range(40,60)) + list(range(80,100)) + list(range(120,180))
    g_step = list(range(10,20)) + list(range(30,40)) + list(range(60,80)) + list(range(100,120)) + list(range(180,200))

    for i, (image, label, MIdex_s, MIdex_t) in enumerate(dataload):
        image, label = image.to(device), label.to(device)
        MIdex_s, MIdex_t = MIdex_s.to(device), MIdex_t.to(device)
        n_batch, nx, ny = image.shape[0], image.shape[2], image.shape[3]
        MIdex_source, MIdex_target = MIdex_s.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, nx, ny), MIdex_t.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, nx, ny)      
        #n_batch, nx, ny, nz = image.shape[0], image.shape[2], image.shape[3], image.shape[4]
        #MIdex_source, MIdex_target = MIdex_s.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, nx, ny, nz), MIdex_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, nx, ny, nz)
        for param_group in S_optimizer.param_groups:
            param_group['lr'] = flearning_rate
        S_net.train()        
        S_optimizer.zero_grad()
        
        G_net.eval()
        G_output = G_net(image, MIdex_target, MIdex_source)
        out_Timg, out_Simg = G_output

        out_Tseg, out_Sseg, out_Sseg_ori = S_net(out_Timg), S_net(out_Simg), S_net(image)
        
        L_ce_f, L_Dice_f = F_loss(out_Tseg, label)
        L_ce_r, L_Dice_r = F_loss(out_Sseg, label)
        L_ce_pr, L_Dice_pr = F_loss(out_Sseg_ori, label)
        loss_f = L_ce_f+0.1*L_Dice_f
        loss_r = L_ce_r+0.1*L_Dice_r
        loss_pr = L_ce_pr+0.1*L_Dice_pr
        loss = loss_r + loss_f + loss_pr
 
        f.write(str(loss.item()))
        f.write('\n')
        f1.write(str(loss_f.item()))
        f1.write('\n')
        f2.write(str(loss_r.item()))
        f2.write('\n')
        f3.write(str(loss_pr.item()))
        f3.write('\n')

        if i % 100 == 0:
            if i > 1:
                flearning_rate = flearning_rate * fregular_rate
            print('epoch %d , %d th, S-Net learning_rate:%.10f , loss: %.10f' % (epoch, i, flearning_rate, loss.item()))


    print('epoch %d , %d th, S-Net learning_rate:%.10f , loss: %.10f' % (epoch, i, flearning_rate, loss.item()))   
    strNetSaveName = 'S_net_with_%d.pkl' % epoch
    torch.save(S_net.state_dict(), os.path.join(savedir, strNetSaveName))
    end_time = time.time()
    print('---------------- Train Seg-Net: ' + strNetSaveName + ' , epoch %d cost time : %3.2f ----------------' % (epoch, end_time - start_time))

def Train_Validate_GAN(dataload, G_net, D_net, S_net, epoch, G_optimizer, D_optimizer, savedir):
    start_time = time.time()
    flearning_rate = LEARNING_RATE*(REGULAR_RATE**(epoch//50))
    fregular_rate = 1.0
    f = open(lossfile, 'a')
    f1 = open(lossfile1, 'a')
    f2 = open(lossfile2, 'a')
    f3 = open(lossfile3, 'a')
    f4 = open(lossfile4, 'a')
    f5 = open(lossfile5, 'a')
    f6 = open(lossfile6, 'a')

    d_step = list(range(0,5)) + list(range(10,15)) + list(range(20,25)) + list(range(30,50)) + list(range(70,90))
    g_step = list(range(5,10)) + list(range(15,20)) + list(range(25,30)) + list(range(50,70)) + list(range(90,99))

    for i, (image, label, MIdex_s, MIdex_t) in enumerate(dataload):
        image, label = image.to(device), label.to(device)
        MIdex_s, MIdex_t = MIdex_s.to(device), MIdex_t.to(device)
        n_batch, nx, ny = image.shape[0], image.shape[2], image.shape[3]
        MIdex_source, MIdex_target = MIdex_s.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, nx, ny), MIdex_t.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, nx, ny)      
        #n_batch, nx, ny, nz = image.shape[0], image.shape[2], image.shape[3], image.shape[4]
        #MIdex_source, MIdex_target = MIdex_s.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, nx, ny, nz), MIdex_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, nx, ny, nz)
        for param_group in G_optimizer.param_groups:
            param_group['lr'] = flearning_rate
        G_net.train()        
        G_optimizer.zero_grad()
        G_output = G_net(image, MIdex_target, MIdex_source)
        out_Timg, out_Simg = G_output
        # print(out_Timg.shape)
        # print(image.shape)
        S_net.eval()
        out_Tseg, out_Sseg = S_net(out_Timg), S_net(out_Simg)
        
        L_shape, L_rec, L_novel = F_loss_G(G_output, image, label, out_Sseg, out_Tseg)

        for param_group in D_optimizer.param_groups:
            param_group['lr'] = flearning_rate
        D_net.eval()
        D_optimizer.zero_grad()
        D_output_r = D_net(image) 
        D_output_f = D_net(out_Timg)
        
        lambda_gp = 10
        lambda_cls = 10 #10
        lambda_adv = 1

        lambda_rec = 10 #100
        lambda_seg = 1 #100
        lambda_shape = epoch//30
        lamdbd_novel = 1

        L_adv_G, disc_cost, L_cls_r, L_cls_f = F_loss_D(D_output_r, D_output_f, MIdex_s, MIdex_t)
        gradient_penalty = F_gradient_penalty(D_net, image, out_Timg)       
        L_adv_D = disc_cost + lambda_gp*gradient_penalty       
        D_loss = lambda_adv*L_adv_D + lambda_cls*L_cls_r        
        G_loss = lambda_adv*L_adv_G  + lambda_cls*L_cls_f + lambda_shape*(L_shape) + lambda_rec*L_rec + lamdbd_novel*L_novel

        G_loss.backward(retain_graph=True)
        G_optimizer.step()
        
        # if i in g_step:
        #     G_loss.backward(retain_graph=True)
        #     G_optimizer.step()
        # else:
        #     D_loss.backward(retain_graph=True)
        #     D_optimizer.step()
        
        f.write(str(L_shape.item()))
        f.write('\n')
        f1.write(str(L_rec.item()))
        f1.write('\n')
        f2.write(str(L_novel.item()))
        f2.write('\n')
        f3.write(str(L_adv_G.item()))
        f3.write('\n')        
        f4.write(str(L_adv_D.item()))
        f4.write('\n')
        f5.write(str(L_cls_r.item()))
        f5.write('\n')
        f6.write(str(L_cls_f.item()))
        f6.write('\n')

        if i % 200 == 0:
            if i > 1:
                flearning_rate = flearning_rate * fregular_rate
            print('epoch %d , %d th, G-Net learning_rate:%.10f , loss: %.10f' % (epoch, i, flearning_rate, G_loss.item()))
            print('epoch %d , %d th, D-Net learning_rate:%.10f , loss: %.10f' % (epoch, i, flearning_rate, D_loss.item()))

    print('epoch %d , %d th, G-Net learning_rate:%.10f , loss: %.10f' % (epoch, i, flearning_rate, G_loss.item()))
    print('epoch %d , %d th, D-Net learning_rate:%.10f , loss: %.10f' % (epoch, i, flearning_rate, D_loss.item()))   
    strNetSaveName = 'G_net_with_%d.pkl' % epoch
    torch.save(G_net.state_dict(), os.path.join(savedir, strNetSaveName))
    strNetSaveName = 'D_net_with_%d.pkl' % epoch
    torch.save(D_net.state_dict(), os.path.join(savedir, strNetSaveName))
    end_time = time.time()
    print('---------------- Train Seg-Net: ' + strNetSaveName + ' , epoch %d cost time : %3.2f ----------------' % (epoch, end_time - start_time))

def main():
    is_for_training = False

    DeepLabV3Plus2d = smp.DeepLabV3Plus('efficientnet-b6', in_channels=1, classes=1, activation='sigmoid')
    Seg_net = DeepLabV3Plus2d.to(device) 
    Seg_net_param = TRAIN_SAVE_DIR_best + 'net_with_99_DL3plus.pkl'
    F_LoadParam(Seg_net_param, Seg_net)
    # for parameter in Seg_net.parameters(): 
    #     parameter.requires_grad = False 

    if is_for_training:       
        G_net = Generator2d(1+n_modality, 1).to(device)
        D_net = Discriminator2d(n_modality).to(device)
        dataset = TrainingDataset(TRAIN_DIR_PATH)
        data_loader = DataLoader(dataset, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM, pin_memory=True)

        cudnn.benchmark = True #False
        # Seg_net = DataParallel(Seg_net, device_ids=[2, 3])
        # G_net = DataParallel(G_net, device_ids=[2, 3])
        # D_net = DataParallel(D_net, device_ids=[2, 3])
        S_optimizer = optim.Adam(Seg_net.parameters())
        G_optimizer = optim.Adam(G_net.parameters())
        D_optimizer = optim.Adam(D_net.parameters())

        D_net_param = TRAIN_SAVE_DIR_Seg + 'D_net_with_99.pkl' 
        F_LoadParam(D_net_param, D_net)

        G_net_param = TRAIN_SAVE_DIR_Seg + 'G_net_with_99.pkl'
        F_LoadParam(G_net_param, G_net)

        for parameter in G_net.parameters(): 
            parameter.requires_grad = False 

        for parameter in D_net.parameters(): 
            parameter.requires_grad = False 
        
        for epoch in range(NumEPOCH):
            #epoch = epoch + 99
            Train_Validate_Seg(data_loader, G_net, Seg_net, epoch, S_optimizer, TRAIN_SAVE_DIR_Seg)       
            #Train_Validate_GAN(data_loader, G_net, D_net, Seg_net, epoch, G_optimizer, D_optimizer, TRAIN_SAVE_DIR_Seg)
    else:
        str_for_action = 'testing'
        print(str_for_action + ' .... ')
        #G_net = Generator2d(1+n_modality, 1).to(device)
        net_param = TRAIN_SAVE_DIR_Seg + 'S_net_with_99.pkl'
        F_LoadParam(net_param, Seg_net)
        Seg_net.eval()

        datafile = glob.glob(TEST_DIR_PATH + '*')
      
        for subjectid in range(len(datafile)):
            imagename = datafile[subjectid] + '/enhanced.nii.gz'
            labelname = datafile[subjectid] + '/atriumSegImgMO.nii.gz'
            predictlabel = ProcessTestDataset2d(imagename, labelname, Seg_net)
            
            savefold = os.path.join(datafile[subjectid] + '/LA_predict_DL3plus_RST_v02.nii.gz')
            nib.save(predictlabel, savefold)

        print(str_for_action + ' end ')

if __name__ == '__main__':
    main()
