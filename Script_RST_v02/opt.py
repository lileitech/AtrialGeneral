import numpy as np
from torch import nn
import torch
from torch.nn import functional as F

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

#------------------------ot------------------------
class OptimalTransport(nn.Module):
    @staticmethod
    def distance(batch1, batch2, dist_metric='cosine'):
        if dist_metric == 'cosine':
            batch1 = F.normalize(batch1, p=2, dim=1)
            batch2 = F.normalize(batch2, p=2, dim=1)
            a = batch1.shape
            b = len(batch1.shape)           
            dist_mat = 1 - torch.mm(batch1, batch2.t())
            #dist_mat = 1 - torch.matmul(batch1, batch2.permute(0, 1, 3, 2))
        elif dist_metric == 'euclidean':
            m, n = batch1.size(0), batch2.size(0)
            dist_mat = torch.pow(batch1, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(batch2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            dist_mat.addmm_(1, -2, batch1, batch2.t()) # squared euclidean distance
        elif dist_metric == 'fast_euclidean':
            batch1 = batch1.unsqueeze(-2)
            batch2 = batch2.unsqueeze(-3)
            dist_mat = torch.sum((torch.abs(batch1 - batch2))**2, -1)
        else:
            raise ValueError(
                'Unknown cost function: {}. Expected to '
                'be one of [cosine | euclidean]'.format(dist_metric)
            )
        return dist_mat

class SinkhornDivergence(OptimalTransport):
    thre = 1e-3

    def __init__(
        self,
        dist_metric='cosine',
        eps=0.01,
        max_iter=5,
        bp_to_sinkhorn=False
    ):
        super().__init__()
        self.dist_metric = dist_metric
        self.eps = eps
        self.max_iter = max_iter
        self.bp_to_sinkhorn = bp_to_sinkhorn

    def forward(self, x, y):
        # x, y: two batches of data with shape (batch, dim)
        W_xy = self.transport_cost(x, y)
        W_xx = self.transport_cost(x, x)
        W_yy = self.transport_cost(y, y)
        return 2*W_xy - W_xx - W_yy

    def transport_cost(self, x, y, return_pi=False):
        C = self.distance(x, y, dist_metric=self.dist_metric)
        pi = self.sinkhorn_iterate(C, self.eps, self.max_iter, self.thre)
        if not self.bp_to_sinkhorn:
            pi = pi.detach()
        cost = torch.sum(pi * C)
        if return_pi:
            return cost, pi
        return cost

    @staticmethod
    def sinkhorn_iterate(C, eps, max_iter, thre):
        nx, ny = C.shape
        mu = torch.ones(nx, dtype=C.dtype, device=C.device) * (1./nx)
        nu = torch.ones(ny, dtype=C.dtype, device=C.device) * (1./ny)
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        
        def M(_C, _u, _v):
            """Modified cost for logarithmic updates.
            Eq: M_{ij} = (-c_{ij} + u_i + v_j) / epsilon
            """
            return (-_C + _u.unsqueeze(-1) + _v.unsqueeze(-2)) / eps

        real_iter = 0 # check if algorithm terminates before max_iter
        # Sinkhorn iterations
        for i in range(max_iter):
            u0 = u
            u = eps * (torch.log(mu + 1e-8) - torch.logsumexp(M(C, u, v), dim=1)) + u
            v = eps * (torch.log(nu + 1e-8) -torch.logsumexp(M(C, u, v).permute(1, 0), dim=1)) + v
            err = (u - u0).abs().sum()
            real_iter += 1
            if err.item() < thre:
                break
        # Transport plan pi = diag(a)*K*diag(b)
        return torch.exp(M(C, u, v))

class MinibatchEnergyDistance(SinkhornDivergence):
    """
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` 
    and 
    :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(
        self,
        dist_metric='cosine',
        eps=0.01,
        max_iter=5,
        bp_to_sinkhorn=False
    ):
        super().__init__(
            dist_metric=dist_metric,
            eps=eps,
            max_iter=max_iter,
            bp_to_sinkhorn=bp_to_sinkhorn
        )

    def forward(self, x, y):
        x1, x2 = torch.split(x, x.size(0) // 2, dim=0)
        y1, y2 = torch.split(y, y.size(0) // 2, dim=0)
        cost = 0
        cost += self.transport_cost(x1, y1)
        cost += self.transport_cost(x1, y2)
        cost += self.transport_cost(x2, y1)
        cost += self.transport_cost(x2, y2)
        cost -= 2 * self.transport_cost(x1, x2)
        cost -= 2 * self.transport_cost(y1, y2)

        return cost

# if __name__ == '__main__':
#     # example: https://dfdazac.github.io/sinkhorn.html
#     n_points = 5
#     a = np.array([[i, 0] for i in range(n_points)])
#     b = np.array([[i, 1] for i in range(n_points)])
#     x = torch.tensor(a, dtype=torch.float)
#     y = torch.tensor(b, dtype=torch.float)

#     sinkhorn = SinkhornDivergence(dist_metric='euclidean', eps=0.01, max_iter=5)
#     dist, pi = sinkhorn.transport_cost(x, y, True)
#     import pdb
#     pdb.set_trace()

#------------------------mmd------------------------
class MaximumMeanDiscrepancy(nn.Module):

    def __init__(self, kernel_type='rbf', normalize=False):
        super(MaximumMeanDiscrepancy, self).__init__()
        self.kernel_type = kernel_type
        self.normalize = normalize

    def forward(self, x, y):
        # x, y: two batches of data with shape (batch, dim)
        # MMD^2(x, y) = k(x, x') - 2k(x, y) + k(y, y')
        if self.normalize:
            x = F.normalize(x, dim=1)
            y = F.normalize(y, dim=1)
        if self.kernel_type == 'linear':
            return self.linear_mmd(x, y)
        elif self.kernel_type == 'poly':
            return self.poly_mmd(x, y)
        elif self.kernel_type == 'rbf':
            return self.rbf_mmd(x, y)
        else:
            raise NotImplementedError

    def linear_mmd(self, x, y):
        # k(x, y) = x^T y
        k_xx = self.remove_self_distance(torch.mm(x, x.t()))
        k_yy = self.remove_self_distance(torch.mm(y, y.t()))
        k_xy = torch.mm(x, y.t())
        return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

    def poly_mmd(self, x, y, alpha=1., c=2., d=2):
        # k(x, y) = (alpha * x^T y + c)^d
        k_xx = self.remove_self_distance(torch.mm(x, x.t()))
        k_xx = (alpha*k_xx + c).pow(d)
        k_yy = self.remove_self_distance(torch.mm(y, y.t()))
        k_yy = (alpha*k_yy + c).pow(d)
        k_xy = torch.mm(x, y.t())
        k_xy = (alpha*k_xy + c).pow(d)
        return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

    def rbf_mmd(self, x, y):
        # k_xx
        d_xx = self.euclidean_squared_distance(x, x)
        d_xx = self.remove_self_distance(d_xx)
        k_xx = self.rbf_kernel_mixture(d_xx)
        # k_yy
        d_yy = self.euclidean_squared_distance(y, y)
        d_yy = self.remove_self_distance(d_yy)
        k_yy = self.rbf_kernel_mixture(d_yy)
        # k_xy
        d_xy = self.euclidean_squared_distance(x, y)
        k_xy = self.rbf_kernel_mixture(d_xy)
        return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

    @staticmethod
    def rbf_kernel_mixture(exponent, sigmas=[1, 5, 10]):
        K = 0
        for sigma in sigmas:
            gamma = 1. / (2. * sigma**2)
            K += torch.exp(-gamma * exponent)
        return K

    @staticmethod
    def remove_self_distance(distmat):
        tmp_list = []
        for i, row in enumerate(distmat):
            row1 = torch.cat([row[:i], row[i + 1:]])
            tmp_list.append(row1)
        return torch.stack(tmp_list)

    @staticmethod
    def euclidean_squared_distance(x, y):
        m, n = x.size(0), y.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, x, y.t())
        return distmat

# #An example to show how to use MaximumMeanDiscrepancy
# mmd = MaximumMeanDiscrepancy(kernel_type='rbf')
# input1, input2 = torch.rand(3, 100), torch.rand(3, 100)
# d = mmd(input1, input2)
# print(d.item())


