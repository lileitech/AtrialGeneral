import numpy as np
from torch import nn
import torch
import collections
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import os
import random
from opt import SinkhornDivergence, MaximumMeanDiscrepancy

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def F_BCEDiceloss(output, label):

    lossfunc = nn.BCELoss().to(device)
    loss_CE = lossfunc(output, label)
    Dice = LabelDice(output, label, [0, 1])
    loss_Dice = 10*(1-torch.mean(Dice[:, 1])) + 0.1*(1-torch.mean(Dice[:, 0]))

    return loss_CE + 0.01*loss_Dice

def F_loss(output, image, label):
    out_seg = output
    
    L_seg = F_BCEDiceloss(out_seg, label)

    return L_seg

def F_loss(output, image, label, dist_type = 'MI'):
    out_seg, out_img, Feature_lab, Feature_img = output
    
    L_seg = F_BCEDiceloss(out_seg, label)
    L_rec = torch.mean(abs(out_img - image))
    
    if dist_type == 'Sinkhorn':
        F_sinkhorn = SinkhornDivergence()
        # F_lab = [Feature_lab[i].flatten(1) for i in range(5)]
        # F_img = [Feature_img[i].flatten(1) for i in range(5)]
        # dist = [F_sinkhorn(F_lab[i], F_img[i]) for i in range(5)]
        # L_dist = -torch.mean(torch.stack(dist))
        L_dist = -F_sinkhorn(Feature_lab[5].flatten(1), Feature_img[5].flatten(1))        
    elif dist_type == 'MMD':
        F_mmd = MaximumMeanDiscrepancy()
        # F_lab = [Feature_lab[i].flatten(1) for i in range(5)]
        # F_img = [Feature_img[i].flatten(1) for i in range(5)]
        # dist = [F_mmd(F_lab[i], F_img[i]) for i in range(5)]
        # L_dist = 1e4*torch.mean(torch.stack(dist))
        L_dist = F_mmd(Feature_lab[5].flatten(1), Feature_img[5].flatten(1))   
    elif dist_type == 'MI':
        from MutualInformation import MutualInformation
        X1, Y1 = Feature_lab[-1], Feature_img[-1] #576
        #X4, Y4 = Feature_lab[-4], Feature_img[-4] #40
        MI1 = MutualInformation(num_bins=X1.shape[1], sigma=0.4, normalize=True).to(device)
        #MI4 = MutualInformation(num_bins=X4.shape[1], sigma=0.4, normalize=True).to(device)      
        visualize_and_save = False
        if visualize_and_save == True:    
            from matplotlib import pyplot as plt
            plt.subplot(121)
            plt.imshow(X[0, 256, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
            plt.subplot(122)
            plt.imshow(Y[0, 256, :, :].cpu().detach().numpy(), cmap=plt.cm.gray) 
            plt.show()
            plt.savefig('/home/lilei/Workspace/AtrialGeneral2021/output_img.jpg')

        L_dist = torch.mean(MI1(X1, Y1)) #+ torch.mean(MI4(X4, Y4))      
        #print(L_dist)
    elif dist_type == 'MINE':
        from mine import MINE
        X, Y = Feature_lab[-1], Feature_img[-1]
        dim = Y.nelement()//Y.shape[0]
        miEstimator = MINE(dim, archSpecs={'layerSizes': [32] * 1,'activationFunctions': ['relu'] * 1}, divergenceMeasure='KL', learningRate=1e-3)       
        idx = torch.randperm(Y.nelement())
        Y_Marginal = Y.view(-1)[idx].view(Y.size())
        L_dist = miEstimator.calcMI(X, Y, X, Y_Marginal, numEpochs=20)   

    return L_seg, L_rec, L_dist


def LabelDice(A, B, class_labels):
    '''
    :param A: (n_batch, 1, n_1, ..., n_k)
    :param B: (n_batch, 1, n_1, ..., n_k)
    :param class_labels: list[n_class]
    :return: (n_batch, n_class)
    '''
    return Dice(torch.cat([1 - torch.clamp(torch.abs(A - i), 0, 1) for i in class_labels], 1),
                torch.cat([1 - torch.clamp(torch.abs(B - i), 0, 1) for i in class_labels], 1))

def Dice(A, B):
    '''
    A: (n_batch, n_class, n_1, n_2, ..., n_k)
    B: (n_batch, n_class, n_1, n_2, ..., n_k)
    return: (n_batch, n_class)
    '''
    eps = 1e-8
#    assert torch.sum(A * (1 - A)).abs().item() < eps and torch.sum(B * (1 - B)).abs().item() < eps
    A = A.flatten(2).float(); B = B.flatten(2).float()
    ABsum = A.sum(-1) + B.sum(-1)
    return 2 * torch.sum(A * B, -1) / (ABsum + eps)


#-----------------load net param-----------------------------
def F_LoadsubParam(net_param, sub_net, target_net):
    print(net_param)
    state_dict = torch.load(net_param, map_location='cpu')
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    sub_net.load_state_dict(new_state_dict)

    # ---------------load the param of Seg_net into SSM_net---------------
    sourceDict = sub_net.state_dict()
    targetDict = target_net.state_dict()
    target_net.load_state_dict({k: sourceDict[k] if k in sourceDict else targetDict[k] for k in targetDict})

def F_LoadParam(net_param, target_net):
    print(net_param)
    state_dict = torch.load(net_param, map_location='cpu')
    target_net.load_state_dict(state_dict)

def F_LoadParam_test(net_param, target_net):
    print(net_param)
    state_dict = torch.load(net_param, map_location='cpu')

    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    target_net.load_state_dict(new_state_dict)
