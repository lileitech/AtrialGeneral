import numpy as np
import nibabel as nib
import torch
from torch import nn
from scipy import stats
from torchvision import transforms
from skimage.transform import rescale
from skimage.exposure import match_histograms

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

height = depth = 192 #208
length = 44 #64, 80
patch_size = (height, depth, length)

class RandomScaling(object):
    """
    Crop randomly scale the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, image, label):
        scale = np.random.randint(3, 5) #scale=(0.75, 1.25)
        image_new = rescale(image, 0.25*scale, anti_aliasing=True)
        label_new = rescale(label, 0.25*scale, anti_aliasing=True)

        return image_new, label_new

class ImageCenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image, label):

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label_new = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image_new = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return image_new, label_new

class LabelCenterCrop(object):
    def __init__(self, output_size=patch_size):
        self.output_size = output_size

    def __call__(self, image, label):

        center_label = label[:, :, int(label.shape[2]/2)]
        center_coord = np.floor(np.mean(np.stack(np.where(center_label > 0)), -1)).astype(np.int16)      
        center_x, center_y = center_coord

        image_new = F_nifity_imageCrop(image, center_coord) 
        label_new = F_nifity_imageCrop(label, center_coord)

        return image_new, label_new

class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, image):
        # noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        # noise = noise + self.mu
        # image_new = image + noise
        image = random_noise(image, mode='gaussian', seed=None, clip=True)        
        #image = exposure.adjust_gamma(image, (0.5+np.random.rand(1)))
        image = img_as_ubyte(image)   
        return image

class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, image, label):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image_new = np.flip(image, axis=axis).copy()
        label_new = np.flip(label, axis=axis).copy()

        return image_new, label_new

class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image, label):

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label_new = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image_new = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return image_new, label_new

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image, label):
        
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        return {torch.from_numpy(image), torch.from_numpy(label).long()}

def F_nifity_imageCrop(numpyimage, center_coord):
    center_x, center_y = center_coord
    shape = numpyimage.shape
    numpyimagecrop = np.zeros((height, depth, shape[2]), dtype=np.float32)
    numpyimagecrop[0:height, 0:depth, :] = \
        numpyimage[int(center_x - height/ 2):int(center_x + height/ 2),
        int(center_y - depth/ 2):int(center_y + depth / 2), :]
    if numpyimage.shape[2] == length:
        numpyimagecrop_new = numpyimagecrop
    elif numpyimage.shape[2] > length:
        numpyimagecrop_new = numpyimagecrop[:, :, (numpyimage.shape[2]-length):numpyimage.shape[2]]
    else:
        pad_width = ((0, 0), (0, 0), (0, int((length - numpyimage.shape[2]))))
        numpyimagecrop_new = np.pad(numpyimagecrop, pad_width, 'constant')

    return numpyimagecrop_new

def Loadimage(imagename):
 
    nibimage= nib.load(imagename)
    imagedata = nibimage.get_data()
    numpyimage = np.array(imagedata).squeeze()   

    return numpyimage

def DataPreprocessing3d(numpyimage, numpylabel):
    
    #numpyimage, numpylabel = RandomRotFlip()(numpyimage, numpylabel) 
    #numpyimage, numpylabel = RandomScaling()(numpyimage, numpylabel)             
    numpyimage, numpylabel = LabelCenterCrop()(numpyimage, numpylabel)  
    #numpyimage = RandomNoise()(numpyimage)

    numpyimage = np.nan_to_num(stats.zscore(numpyimage))

    return numpyimage, numpylabel

def LoadDataset3d(imagenames, labelnames, DA = True, hist_matching = False):

    niblabel = nib.load(labelnames)
    labeldata = niblabel.get_data()
    numpylabel = np.array(labeldata).squeeze()      
    nibimage = nib.load(imagenames)
    imagedata = nibimage.get_data()
    numpyimage = np.array(imagedata).squeeze()  
    if DA == False:
        numpyimage_new, numpylabel_new = LabelCenterCrop()(numpyimage, numpylabel)  
        numpyimage_new = np.nan_to_num(stats.zscore(numpyimage_new))
    else:
        numpyimage_new, numpylabel_new = DataPreprocessing3d(numpyimage, numpylabel)   
    if hist_matching == True:
        refimagename = '/home/lilei/Workspace/AtrialGeneral2021/Data/IC_1/Post/train_data/P01035_6/'
        numpyimage_ref = Loadimage(refimagename + 'enhanced.nii.gz')
        numpylabel_ref = Loadimage(refimagename + 'atriumSegImgMO.nii.gz')
        numpyimage_ref_new, _ = DataPreprocessing(numpyimage_ref, numpylabel_ref)
        numpyimage_new_matched = match_histograms(numpyimage_new, numpyimage_ref_new, multichannel=True)                       
        visual_index = False
        if visual_index == True:
            from matplotlib import pyplot as plt
            plt.figure()
            plt.subplot(321)
            plt.imshow(numpyimage[:, :, 20], cmap=plt.cm.gray)
            plt.subplot(322)
            plt.imshow(numpylabel[:, :, 20], cmap=plt.cm.gray)
            plt.subplot(323)
            plt.imshow(numpyimage_new[:, :, 20], cmap=plt.cm.gray)
            plt.subplot(324)
            plt.imshow(numpylabel_new[:, :, 20], cmap=plt.cm.gray)
            plt.subplot(325)
            plt.imshow(numpyimage_ref_new[:, :, 20], cmap=plt.cm.gray)
            plt.subplot(326)
            plt.imshow(numpyimage_new_matched[:, :, 20], cmap=plt.cm.gray)
            plt.savefig('output_img.jpg')
        numpyimage_new = numpyimage_new_matched
    return np.expand_dims(numpyimage_new, 0), np.expand_dims(numpylabel_new, 0)

def LoadDataset2d(imagename, labelname, DA = True, hist_matching = False):
    numpy2Dimage = []
    numpy2Dlabel = []
    NumSlice = 0
    #print('loading training image: ' + imagename)

    niblabel = nib.load(labelname)
    labeldata = niblabel.get_data()
    numpylabel = np.array(labeldata).squeeze()
    nibimage = nib.load(imagename)
    imagedata = nibimage.get_data()
    numpyimage = np.array(imagedata).squeeze()

    # center_numpylabel = numpylabel[:, :, int(numpylabel.shape[2] / 2)]
    # center_coord = np.floor(np.mean(np.stack(np.where(center_numpylabel > 0)), -1)).astype(np.int16)
    # numpylabel_new = F_nifity_imageCrop(numpylabel, center_coord)
    # numpyimage_new = F_nifity_imageCrop(numpyimage, center_coord) 

    if DA == False:        
        numpyimage_new, numpylabel_new = LabelCenterCrop()(numpyimage, numpylabel)  
        numpyimage_new = np.nan_to_num(stats.zscore(numpyimage_new))
    else:
        numpyimage_new, numpylabel_new = DataPreprocessing3d(numpyimage, numpylabel) 
    if hist_matching == True:
        refimagename = '/home/lilei/Workspace/AtrialGeneral2021/Data/IC/Post/train_data/P01035_6/'
        numpyimage_ref = Loadimage(refimagename + 'enhanced.nii.gz')
        numpylabel_ref = Loadimage(refimagename + 'atriumSegImgMO.nii.gz')
        numpyimage_ref_new, _ = DataPreprocessing3d(numpyimage_ref, numpylabel_ref) 
        numpyimage_new_matched = match_histograms(numpyimage_new, numpyimage_ref_new, multichannel=True)                       
        visual_index = False
        if visual_index == True:
            from matplotlib import pyplot as plt
            plt.figure()
            plt.subplot(321)
            plt.imshow(numpyimage[:, :, 20], cmap=plt.cm.gray)
            plt.subplot(322)
            plt.imshow(numpylabel[:, :, 20], cmap=plt.cm.gray)
            plt.subplot(323)
            plt.imshow(numpyimage_new[:, :, 20], cmap=plt.cm.gray)
            plt.subplot(324)
            plt.imshow(numpylabel_new[:, :, 20], cmap=plt.cm.gray)
            plt.subplot(325)
            plt.imshow(numpyimage_ref_new[:, :, 20], cmap=plt.cm.gray)
            plt.subplot(326)
            plt.imshow(numpyimage_new_matched[:, :, 20], cmap=plt.cm.gray)
            plt.savefig('output_img.jpg')
        numpyimage_new = numpyimage_new_matched

    size = numpyimage_new.shape
    for sliceid in range(0, size[2]):
        numpy2Dimage.append(numpyimage_new[:, :, sliceid])
        numpy2Dlabel.append(numpylabel_new[:, :, sliceid])
        NumSlice = NumSlice + 1

    return numpy2Dimage, numpy2Dlabel, NumSlice

def ProcessTestDataset2d(imagename, labelname, net):

    print('loading test image: ' + imagename)
    nibimage = nib.load(imagename)
    shape = nibimage.shape
    numpyimage, numpylabel, NumSlice = LoadDataset2d(imagename, labelname, DA = False)
   
    numpylabel_ori = Loadimage(labelname)
    center_label = numpylabel_ori[:, :, int(numpylabel_ori.shape[2]/2)]
    crop_center_coord = np.floor(np.mean(np.stack(np.where(center_label > 0)), -1)).astype(np.int16)      
  

    for sliceid in range(NumSlice):
        tensorimage = torch.from_numpy(np.array([numpyimage[sliceid]])).unsqueeze(0).float().to(device)
        tensorlabel = torch.from_numpy(np.array([numpylabel[sliceid]])).unsqueeze(0).float().to(device)      
        #MIdex_t, MIdex_s = torch.zeros_like(tensorimage), torch.zeros_like(tensorimage)
        #MIdex_source, MIdex_target = MIdex_s.repeat(1, 3, 1, 1).to(device), MIdex_t.repeat(1, 3, 1, 1).to(device)
        output = net(tensorimage)
        #out_Tseg, out_Timg, out_Sseg, out_Simg = output

        output = output.squeeze().cpu().detach().numpy()
        outputlab = (output > 0.5) * 1

        outputlab = outputlab[:, :, np.newaxis]
        if sliceid == 0:
            label = outputlab
        else:
            label = np.concatenate((label, outputlab), axis=-1)

    center_x, center_y = crop_center_coord
    pad_width = ((int(center_x - height//2),int(shape[0] - center_x - height//2)),(int(center_y - depth//2),int(shape[1] - center_y - depth//2)), (0,0))
    predictnumpylabel = np.pad(label, pad_width, 'constant')
    predictlabel = nib.Nifti1Image(predictnumpylabel, nibimage.affine, nibimage.header)

    return predictlabel
