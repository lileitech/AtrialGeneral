import nibabel as nib
import numpy as np
import os
import glob
import math
import torch
import segmentation_models_pytorch as smp
from function import F_LoadParam
from loaddata import LoadDataset3d
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

datasetname = '*_1' #IC, Utah_I, KCL, Yale, *_1
post_pre = '*' #Post, Pre
Root_DIR = '/home/lilei/Workspace/AtrialGeneral2021/'

TRAIN_SAVE_DIR_Seg = Root_DIR + 'Script_Unet3d/result_model/'
TRAIN_SAVE_DIR_best = Root_DIR + 'Script_Unet3d/best_model/'


height = depth = 256 #208, 192
length = 44 #64, 80
patch_size = (height, depth, length)

DeepLabV3Plus2d = smp.DeepLabV3Plus('efficientnet-b6', in_channels=1, classes=1, activation='sigmoid')
Seg_net = DeepLabV3Plus2d.to(device)
Seg_net_param = TRAIN_SAVE_DIR_best + 'net_with_99_all_DL3plus_base.pkl'
F_LoadParam(Seg_net_param, Seg_net)
Seg_net.eval()

def Loadimage(imagename):
 
    nibimage= nib.load(imagename)
    imagedata = nibimage.get_fdata()
    numpyimage = np.array(imagedata).squeeze()   

    return numpyimage, nibimage

def Getfeature(Rootdir):
    imagename = Rootdir + 'enhanced.nii.gz'
    labelname = Rootdir + 'atriumSegImgMO.nii.gz'
    print(imagename)
    numpyimage, numpylabel = LoadDataset3d(imagename, labelname)
    # numpyimage, _ = Loadimage(imagename)
    # numpylabel, _ = Loadimage(labelname)
    tensorimage, tensorlabel = torch.from_numpy(numpyimage).float().unsqueeze(0), torch.from_numpy(numpylabel).float().unsqueeze(0)
    tensorimage2d, tensorlabel2d = tensorimage[:, :, :, :, 20].to(device), tensorlabel[:, :, :, :, 20].to(device)
    
    E_feature = Seg_net.encoder(tensorimage2d)
    features = E_feature[4].squeeze().cpu().detach().numpy()

    return features.reshape(features.shape[0], -1)

features_A = Getfeature(Root_DIR + 'Data/Utah_I/Post/train_data/patient_8/')
features_B = Getfeature(Root_DIR + 'Data/IC/Post/train_data/P02008d_6/')
features_C = Getfeature(Root_DIR + 'Data/KCL_1/Post/train_data/p1/')
features_D = Getfeature(Root_DIR + 'Data/Yale_1/Post/train_data/p3/')

# Y = TSNE(n_components=2).fit_transform(X)
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(2, 1, 2)
# plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)

# plt.show()
# plt.savefig(Root_DIR + "/test.jpg")

tsne = TSNE()
n = 1
# X_embedded = tsne.fit_transform(np.concatenate((features_A[n, :], features_B[n, :], features_C[n, :], features_D[n, :]), axis=0))
# Y = ['Utah']*features_A[n, :].shape[0]+['IC']*features_A[n, :].shape[0]+['KCL']*features_C[n, :].shape[0]+['Yale']*features_D[n, :].shape[0]
X_embedded = tsne.fit_transform(np.concatenate((features_A, features_B, features_C, features_D), axis=0))
Y = ['Utah']*features_A.shape[0]+['IC']*features_A.shape[0]+['KCL']*features_C.shape[0]+['Yale']*features_D.shape[0]
sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=Y, legend='full', palette="deep")

plt.show()
plt.savefig(Root_DIR + "/test.jpg")
plt.close()