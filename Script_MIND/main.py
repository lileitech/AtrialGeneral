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

from loaddata import LoadDataset2d, LoadDataset3d, ProcessTestDataset3d, ProcessTestDataset2d
from network import MIDNet3d, MIDNet2d
from function import F_LoadParam_test, F_LoadParam, F_loss

datasetname = '*_1' #IC, Utah_I, KCL, Yale, *_1
post_pre = '*' #Post, Pre
Root_DIR = '/home/lilei/Workspace/AtrialGeneral2021/'

Data_DIR = Root_DIR + 'Data/' + datasetname + '/' + post_pre + '/'
TRAIN_DIR_PATH = Data_DIR + 'train_data/'
#TEST_DIR_PATH = Data_DIR + 'test_data/'
TEST_DIR_PATH = Root_DIR + 'Data/Utah_I/' + post_pre + '/test_data/'
TRAIN_SAVE_DIR_Seg = Root_DIR + 'Script_MIND/result_model/'
TRAIN_SAVE_DIR_best = Root_DIR + 'Script_MIND/best_model/'

lossfile_DIR = Root_DIR + 'Script_MIND/lossfile/'
Seglossfile = lossfile_DIR + 'L_seg.txt'
imglossfile1 = lossfile_DIR + 'L_rec.txt'
imglossfile2 = lossfile_DIR + 'L_dist.txt'

WORKERSNUM = 16 #16
BatchSize = 10 #30, 50
NumEPOCH = 100 #250
LEARNING_RATE = 1e-4#3e-4
REGULAR_RATE = 0.95
WEIGHT_DECAY = 1e-4
n_modality = 3

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
testonecase = False

class TrainingDataset(data.Dataset):
    def __init__(self, datapath):
        self.datafile = glob.glob(datapath + '*')
        self.numpyimage = []
        self.numpylabel = []
        self.NumOfSubjects = 0

        for subjectid in range(len(self.datafile)): #2250 slices
            if testonecase == True:
                if subjectid > 0:
                    break
            imagename = self.datafile[subjectid] + '/enhanced.nii.gz'
            labelname = self.datafile[subjectid] + '/atriumSegImgMO.nii.gz'

            print('loading training image: ' + imagename)
            numpyimage, numpylabel, NumSlice = LoadDataset2d(imagename, labelname)
            
            self.numpyimage.extend(numpyimage)
            self.numpylabel.extend(numpylabel)
            self.NumOfSubjects += NumSlice 
            
    def __getitem__(self, item):

        numpyimage = np.array([self.numpyimage[item]])
        numpylabel = np.array([self.numpylabel[item]])
        numpylabel = (numpylabel > 0) * 1
 
        tensorimage = torch.from_numpy(numpyimage).float()
        tensorlabel = torch.from_numpy(numpylabel.astype(np.float32))

        return tensorimage, tensorlabel

    def __len__(self):
        return self.NumOfSubjects

def Train_Validate(dataload, net, epoch, optimizer, savedir):
    start_time = time.time()
    flearning_rate = LEARNING_RATE*(REGULAR_RATE**(epoch//50))
    fregular_rate = 1.0
    f = open(Seglossfile, 'a')
    f1 = open(imglossfile1, 'a')
    f2 = open(imglossfile2, 'a')

    net.train()
    for i, (image, label) in enumerate(dataload):
        for param_group in optimizer.param_groups:
            param_group['lr'] = flearning_rate
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()

        output = net(image)
        L_seg = F_loss(output, image, label)
        L_seg, L_rec, L_dist = F_loss(output, image, label)
        if epoch>20:
            lambda_dist = 0.01
        else:
            lambda_dist = 0
        loss = 10*L_seg + 0.01*L_rec + 0.01*L_dist

        loss.backward()
        optimizer.step()

        f.write(str(L_seg.item()))
        f.write('\n')

        f1.write(str(L_rec.item()))
        f1.write('\n')

        f2.write(str(L_dist.item()))
        f2.write('\n')

        if i % 50 == 0:
            print('epoch %d , %d th, Seg-Net learning_rate:%.10f , loss: %.10f' % (epoch, i, flearning_rate, loss.item()))

    print('epoch %d , %d th, Seg-Net learning_rate:%.10f , loss: %.10f' % (epoch, i, flearning_rate, loss.item()))
    strNetSaveName = 'net_with_%d.pkl' % epoch
    torch.save(net.state_dict(), os.path.join(savedir, strNetSaveName))
    end_time = time.time()
    print('---------------- Train Seg-Net: ' + strNetSaveName + ' , epoch %d cost time : %3.2f ----------------' % (epoch, end_time - start_time))

def main():
    is_for_training = False
    
    if is_for_training:
        net = MIDNet2d('efficientnet-b6', in_channels=1, classes=1).to(device)
        dataset = TrainingDataset(TRAIN_DIR_PATH)
        data_loader = DataLoader(dataset, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM, pin_memory=True)
        cudnn.benchmark = True
        #net = DataParallel(net, device_ids=[1, 2])

        #optimizer = optim.Adam(net.parameters())
        optimizer = optim.SGD(net.parameters(), LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

        for epoch in range(NumEPOCH):
            Train_Validate(data_loader, net, epoch, optimizer, TRAIN_SAVE_DIR_Seg)

    else:
        str_for_action = 'testing'
        print(str_for_action + ' .... ')
        Seg_net = MIDNet2d('efficientnet-b6', in_channels=1, classes=1).to(device)
        Seg_net_param = TRAIN_SAVE_DIR_Seg + 'net_with_99.pkl'
        F_LoadParam(Seg_net_param, Seg_net)
        Seg_net.eval()

        datafile = glob.glob(TEST_DIR_PATH + '/*')

        for subjectid in range(len(datafile)):
            imagename = datafile[subjectid] + '/enhanced.nii.gz'
            labelname = datafile[subjectid] + '/atriumSegImgMO.nii.gz'
            predictlabel = ProcessTestDataset2d(imagename, labelname, Seg_net)

            savefold = os.path.join(datafile[subjectid] + '/LA_predict_DL3plus_MIND.nii.gz')
            nib.save(predictlabel, savefold)

        print(str_for_action + ' end ')

if __name__ == '__main__':
    main()
