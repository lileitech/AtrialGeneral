import numpy as np
from torch import nn
import torch
import collections
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def F_loss_scar(output, label_LA, label_scar):

    out_LA, out_scar = output
    lossfunc1 = nn.BCELoss().to(device)
    loss_la = lossfunc1(out_LA, label_LA)
    loss_scar = lossfunc1(out_scar, label_scar)

    return loss_la, loss_scar

def F_loss(output, label):

    lossfunc = nn.BCELoss().to(device)
    CE_loss = lossfunc(output, label)
    Dice = LabelDice(output, label, [0, 1])
    weighted_Dice_loss = 10*(1-torch.mean(Dice[:, 1])) + 0.1*(1-torch.mean(Dice[:, 0]))

    return CE_loss, weighted_Dice_loss

def F_loss_SDM(output, label):
    lossfunc = nn.BCELoss().to(device)
    CE_loss = lossfunc(output, label)
    loss_seg = CE_loss

    gt_dis = compute_sdf(label.cpu().numpy(), output.shape)
    gt_dis = torch.from_numpy(gt_dis).float().to(device)
    loss_sdf_lei = torch.mean(((output - 0.5) * gt_dis))

    return loss_seg, loss_sdf_lei

def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """
    T = 50
    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = T*np.ones(out_shape) #np.zeros(out_shape)
    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                #sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                sdf = negdis - posdis
                sdf[boundary==1] = 0
                normalized_sdf[b][c] = sdf
                # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
                # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return np.clip(normalized_sdf, -T, T)


def LabelDice(A, B, class_labels):
    '''
    :param A: (n_batch, 1, n_1, ..., n_k)
    :param B: (n_batch, 1, n_1, ..., n_k)
    :param class_labels: list[n_class]
    :return: (n_batch, n_class)
    '''
    return F_Dice(torch.cat([1 - torch.clamp(torch.abs(A - i), 0, 1) for i in class_labels], 1),
                torch.cat([1 - torch.clamp(torch.abs(B - i), 0, 1) for i in class_labels], 1))

def F_Dice(A, B):
    '''
    A: (n_batch, n_class, ...)
    B: (n_batch, n_class, ...)
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
