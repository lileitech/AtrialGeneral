import numpy as np
from torch import nn
import torch
import collections
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import os
from opt import MinibatchEnergyDistance, SinkhornDivergence, MaximumMeanDiscrepancy

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().to(device)
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().to(device)

        u = torch.zeros_like(mu).to(device)
        v = torch.zeros_like(nu).to(device)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost #, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

def F_mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   
		os.makedirs(path)           

def F_BCEDiceloss(output, label):

    lossfunc = nn.BCELoss().to(device)
    loss_CE = lossfunc(output, label)
    Dice = LabelDice(output, label, [0, 1])
    loss_Dice = 10*(1-torch.mean(Dice[:, 1])) + 0.1*(1-torch.mean(Dice[:, 0]))

    return loss_CE + 0.01*loss_Dice

def F_loss(output, label):

    lossfunc = nn.BCELoss().to(device)
    CE_loss = lossfunc(output, label)
    Dice = LabelDice(output, label, [0, 1])
    weighted_Dice_loss = 10*(1-torch.mean(Dice[:, 1])) + 0.1*(1-torch.mean(Dice[:, 0]))

    return CE_loss, weighted_Dice_loss

def F_loss_rec(output, image, label):
    out_seg, out_img = output
    
    lossfunc1 = nn.CrossEntropyLoss().to(device)
    lossfunc2 = nn.L1Loss().to(device)
    label = label.squeeze(1)
    L_seg = lossfunc1(out_seg, label)
    L_rec = torch.mean(abs(out_img - image))

    return L_seg, L_rec

def F_loss_G(output, image, label, out_Sseg, out_Tseg, n_modality = 3):
    out_Timg, out_Simg = output
    
    lossfunc1 = nn.L1Loss().to(device)  
    L_shape = torch.mean(abs(out_Sseg - out_Tseg))
    L_rec = torch.mean(abs(out_Simg - image))
    
    #F_sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction='mean')
    L_novel = F_loss_diversity(image.flatten(1), out_Timg.flatten(1)) 
    #L_novel = L_rec

    visual_index = False   
    if visual_index == True:
        from matplotlib import pyplot as plt
        plt.figure()
        size = '2d'
        if size == '2d':
            plt.subplot(231)
            plt.imshow(out_Tseg[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
            plt.subplot(232)
            plt.imshow(out_Sseg[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
            plt.subplot(233)
            plt.imshow(out_Timg[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
            plt.subplot(234)
            plt.imshow(out_Simg[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
            plt.subplot(235)
            plt.imshow(image[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
            plt.subplot(236)
            plt.imshow(label[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
        elif size == '3d':
            plt.subplot(231)
            plt.imshow(out_Tseg[0, 0, :, :, 30].cpu().detach().numpy(), cmap=plt.cm.gray)
            plt.subplot(232)
            plt.imshow(out_Sseg[0, 0, :, :, 30].cpu().detach().numpy(), cmap=plt.cm.gray)
            plt.subplot(233)
            plt.imshow(out_Timg[0, 0, :, :, 30].cpu().detach().numpy(), cmap=plt.cm.gray)
            plt.subplot(234)
            plt.imshow(out_Simg[0, 0, :, :, 30].cpu().detach().numpy(), cmap=plt.cm.gray)
            plt.subplot(235)
            plt.imshow(image[0, 0, :, :, 30].cpu().detach().numpy(), cmap=plt.cm.gray)
            plt.subplot(236)
            plt.imshow(label[0, 0, :, :, 30].cpu().detach().numpy(), cmap=plt.cm.gray)            
        #plt.savefig('output_img.pdf')
        plt.savefig('output_img.jpg')
 
    return L_shape, L_rec, L_novel


def F_loss_D(out_real, out_false, MIdex_s, MIdex_t):

    disc_real, cls_real = out_real
    disc_fake, cls_fake = out_false
    # disc_real = out_real
    # disc_fake = out_false
    gen_cost = -torch.mean(disc_fake)
    disc_cost = torch.mean(disc_fake) - torch.mean(disc_real)

    lossfunc1 = nn.BCELoss().to(device)
    # print(cls_real.shape)
    # print(MIdex_s.shape)
    L_cls_real = torch.mean(abs(cls_real.squeeze() - MIdex_s))#lossfunc1(cls_real, MIdex_s)
    L_cls_fake = torch.mean(abs(cls_fake.squeeze() - MIdex_t))# lossfunc1(cls_fake, MIdex_t)
    
    #L_diversity = L_cls_fake
    #L_diversity = F_loss_diversity(cls_real, cls_fake)

    return gen_cost, disc_cost, L_cls_real, L_cls_fake

def F_loss_diversity(x1, x2, G_maxJintra=10, G_maxJinter=10, G_maxJfakeinter = 10): 
    real_features, fake_features = x1.flatten(1), x2.flatten(1)   
    # split features
    split_batch = real_features.size(0) // 3
    real_features_list = torch.split(real_features, split_batch, dim=0)
    fake_features_list = torch.split(fake_features, split_batch, dim=0)

    # --------------start calculate the domain diversity loss----------
    distribution_divergence = SinkhornDivergence()#MaximumMeanDiscrepancy,SinkhornDivergence
    
    loss_div = 0.
    if G_maxJintra > 0:
        # maximize intra-domain distribution divergence loss
        J_intra = 0.
        count = 0
        for rf, ff in zip(real_features_list, fake_features_list):
            J_intra += -distribution_divergence(rf, ff)
            count += 1
        J_intra = J_intra / count
        loss_div += G_maxJintra * J_intra
    if G_maxJinter > 0:
        # maximize inter-domain distribution divergence loss
        J_inter = 0.
        count = 0
        for i, ff in enumerate(fake_features_list):
            for j, rf in enumerate(real_features_list):
                if j != i:
                    J_inter += -distribution_divergence(rf, ff)
                    count += 1
        J_inter = J_inter / count
        loss_div += G_maxJinter * J_inter
    if G_maxJfakeinter > 0:
        # maximize inter-domain divergence between fake data
        J_fake_inter = 0.
        count = 0
        for i in range(len(fake_features_list)):
            for j in range(i+1, len(fake_features_list)):
                ff_i = fake_features_list[i]
                ff_j = fake_features_list[j]
                J_fake_inter += -distribution_divergence(ff_i, ff_j)
                count += 1
        J_fake_inter = J_fake_inter / count
        loss_div += G_maxJfakeinter * J_fake_inter

    return loss_div

def F_gradient_penalty(D_net, real_data, generated_data):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data).to(device)
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True).to(device)

    # Calculate probability of interpolated examples
    prob_interpolated,_ = D_net(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                            create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)
    #self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return ((gradients_norm - 1) ** 2).mean()


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
