import os
import time
import glob
import torch
import numpy as np
import nibabel as nib
from torch import optim
from torch.nn import DataParallel
from torch.backends import cudnn
import torch.utils.data as data
from torch.utils.data import DataLoader
# from torchvision import transforms
import segmentation_models_pytorch as smp
#from pympler.tracker import SummaryTracker

from loaddata import LoadDataset3d, LoadDataset2d, ProcessTestDataset2d, ProcessTestDataset3d
from function import F_loss, F_loss_scar, F_LoadParam, F_LoadParam_test

datasetname = '*_1' #IC, Utah_I, KCL_1, Yale_1, *_1
post_pre = '*' #Post, Pre
Root_DIR = '/home/lilei/Workspace/AtrialGeneral2021/'

Data_DIR = Root_DIR + 'Data/' + datasetname + '/' + post_pre + '/'
TRAIN_DIR_PATH = Data_DIR + 'train_data/'
#TEST_DIR_PATH = Data_DIR + 'test_data/'
TEST_DIR_PATH = Root_DIR + 'Data/Utah_I/' + post_pre + '/test_data/'
TRAIN_SAVE_DIR_Seg = Root_DIR + 'Script_net/result_model/'
TRAIN_SAVE_DIR_best = Root_DIR + 'Script_net/best_model/'

lossfile_DIR = Root_DIR + 'Script_net/lossfile/'
lossfile = lossfile_DIR + '/loss.txt'
lossfile1 = lossfile_DIR + '/loss_1.txt'
lossfile2 = lossfile_DIR + '/loss_2.txt'

WORKERSNUM = 8 #16
BatchSize = 20 #2
NumEPOCH = 100
LEARNING_RATE = 1e-4 #1e-3
REGULAR_RATE = 0.96
WEIGHT_DECAY = 1e-4

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print('Lei is using ' + str(device))
testonecase = False

class TrainingDataset(data.Dataset):
    def __init__(self, datapath):

        self.numpyimage = []
        self.numpylabel = []
        self.NumOfSubjects = 0

        self.datafile = glob.glob(datapath + '*')
        for subjectid in range(len(self.datafile)):
            if testonecase == True:
                #if subjectid > 1:
                #    break
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
    flearning_rate = LEARNING_RATE*(REGULAR_RATE**(epoch//10))
    if flearning_rate<1e-5:
        flearning_rate = 1e-5
    f = open(lossfile, 'a')    
    f1 = open(lossfile1, 'a')
    f2 = open(lossfile2, 'a')

    net.train()
    for i, (image, label) in enumerate(dataload):
        for param_group in optimizer.param_groups:
            param_group['lr'] = flearning_rate
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()

        output = net(image)
        loss_CE, loss_Dice = F_loss(output, label)
        loss = loss_CE + 0.01*loss_Dice

        loss.backward()
        optimizer.step()

        f.write(str(loss.item()))
        f.write('\n')

        f1.write(str(loss_CE.item()))
        f1.write('\n')

        f2.write(str(loss_Dice.item()))
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

    #'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    #'se_resnet50', 'se_resnet101','se_resnet152'
    #'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7'
    #Unet2d = smp.Unet('efficientnet-b6', in_channels=1, classes=1, activation='sigmoid')
    #UnetPlusPlus2d = smp.UnetPlusPlus('efficientnet-b6', in_channels=1, classes=1, activation='sigmoid')
    #MAnet2d = smp.MAnet('efficientnet-b6', in_channels=1, classes=1, activation='sigmoid')
    DeepLabV3Plus2d = smp.DeepLabV3Plus('efficientnet-b6', in_channels=1, classes=1, activation='sigmoid')

    if is_for_training:
        #net = U_Net3d(1, 1).to(device)
        net = DeepLabV3Plus2d.to(device) #Unet2d, UnetPlusPlus2d, MAnet2d, DeepLabV3Plus2d
        
        dataset = TrainingDataset(TRAIN_DIR_PATH)
        data_loader = DataLoader(dataset, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM, pin_memory=True)
        cudnn.benchmark = True
        #net = DataParallel(net, device_ids=[1, 2])

        optimizer = optim.Adam(net.parameters())
        #optimizer = optim.SGD(net.parameters(), LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

        for epoch in range(NumEPOCH):
            Train_Validate(data_loader, net, epoch, optimizer, TRAIN_SAVE_DIR_Seg)

    else:
        str_for_action = 'testing'
        print(str_for_action + ' .... ')
        Seg_net = DeepLabV3Plus2d.to(device) #Unet2d, UnetPlusPlus2d, MAnet2d, DeepLabV3Plus2d
        Seg_net_param = TRAIN_SAVE_DIR_Seg + 'net_with_99.pkl'
        F_LoadParam(Seg_net_param, Seg_net)
        Seg_net.eval()

        datafile = glob.glob(TEST_DIR_PATH + '/*')

        for subjectid in range(len(datafile)):       
            savefold = os.path.join(datafile[subjectid] + '/LA_predict_DL3plus.nii.gz')
            # savefold = savefold_old.replace('Utah_I', 'Utah_I_result')
            # F_mkdir(savefold)

            imagename = datafile[subjectid] + '/enhanced.nii.gz'
            labelname = datafile[subjectid] + '/atriumSegImgMO.nii.gz'
            predictlabel = ProcessTestDataset2d(imagename, labelname, Seg_net)      
            nib.save(predictlabel, savefold)

        print(str_for_action + ' end ')

if __name__ == '__main__':
    main()
