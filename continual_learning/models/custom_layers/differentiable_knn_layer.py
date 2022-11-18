'''
"Differentiable k-nearest neighbors" layer.

Given a set of M queries and a set of N neighbors,
returns an M x N matrix whose rows sum to k, indicating to what degree
a certain neighbor is one of the k nearest neighbors to the query.
At the limit of tau = 0, each entry is a binary value representing
whether each neighbor is actually one of the k closest to each query.
'''
from collections import defaultdict

# https://github.com/ermongroup/neuralsort/blob/master/pytorch/dknn_layer.py
# https://proceedings.neurips.cc/paper/2020/hash/ec24a54d62ce57ba93a531b460fa8d18-Abstract.html

import torch
from torch.autograd import Function
import torch.nn.functional as F


class DKNN(torch.nn.Module):
    def __init__(self, k, num_samples=-1, num_neighbors=100, use_manual_grad=True, epsilon=0.1,
                 max_iter=200, device: torch.device = torch.device('cpu')):
        super(DKNN, self).__init__()
        self.k = k
        self.n = num_neighbors
        self.device = device
        if use_manual_grad:
            self.soft_sort = TopK_custom(k, epsilon=epsilon, max_iter=max_iter, device=self.device)
        else:
            self.soft_sort = TopK_stablized(k, num_neighbors, epsilon=epsilon, max_iter=max_iter, device=self.device)

        self.num_samples = num_samples
        self.run_meta = defaultdict(int)

    def forward(self, query, neighbors, cosine_distance: bool = True, return_distances: bool = True, tau=1.0):
        if cosine_distance:
            cos_similarity = torch.transpose(torch.matmul(neighbors, query.T), dim0=0, dim1=1)
            scores = 1 - cos_similarity
            distances = cos_similarity
        else:
            diffs = (query.unsqueeze(1) - neighbors.unsqueeze(0))
            squared_diffs = diffs ** 2
            l2_norms = squared_diffs.sum(2)
            norms = l2_norms
            scores = norms
            distances = scores

        P_hat = self.soft_sort(scores)
        top_k_seq = P_hat[:, 0, :] * self.n

        condition = top_k_seq < 0.3
        condition = condition.to(self.device)
        if_yes = torch.tensor(0.0).to(self.device)
        if_no = torch.tensor(1.0).to(self.device)
        mask = torch.where(condition, if_yes, if_no)

        top_k_seq = top_k_seq * mask

        self.run_meta['neighbors_count'] += torch.sum(mask).item()
        self.run_meta['total_samples'] += len(mask)

        if return_distances:
            return top_k_seq, distances, scores

        return top_k_seq


class TopK(torch.nn.Module):
    def __init__(self, k, n, epsilon=0.1, max_iter=200, device: torch.device = torch.device('cpu')):
        super(TopK, self).__init__()
        self.n = n
        self.mu = torch.ones([1, n], requires_grad=False) / n
        self.nu = torch.FloatTensor([k / n, (n - k) / n]).view([1, 2])
        self.epsilon = epsilon
        self.anchors = torch.FloatTensor([0, 1]).view([1, 2, 1])
        self.max_iter = max_iter
        self.device = device

        self.mu = self.mu.to(self.device)
        self.nu = self.nu.to(self.device)
        self.anchors = self.anchors.to(self.device)

    def forward(self, scores):
        batch_size = scores.size(0)
        scores = scores.view([batch_size, 1, -1])
        C = (scores - self.anchors) ** 2
        C = C / C.max()
        G = torch.exp(-C / self.epsilon)
        v = torch.ones([batch_size, self.n])

        v = v.to(self.device)

        pad = 1e-16
        for i in range(self.max_iter):
            u = self.nu / ((G * (v.unsqueeze(-2))).sum(-1) + pad)
            v = self.mu / ((G * (u.unsqueeze(-1))).sum(-2) + pad)

        P = u.unsqueeze(-1) * G * v.unsqueeze(-2)

        return P


class TopK_stablized(torch.nn.Module):
    def __init__(self, k, n, epsilon=0.1, max_iter=200, device: torch.device = torch.device('cpu')):
        super(TopK_stablized, self).__init__()
        self.n = n
        self.mu = torch.ones([1, 1, n], requires_grad=False) / n
        self.nu = torch.FloatTensor([k / n, (n - k) / n]).view([1, 2, 1])
        self.epsilon = epsilon
        self.anchors = torch.FloatTensor([0, 1]).view([1, 2, 1])
        self.max_iter = max_iter
        self.device = device

        self.mu = self.mu.to(self.device)
        self.nu = self.nu.to(self.device)
        self.anchors = self.anchors.to(self.device)

    def forward(self, scores):
        batch_size = scores.size(0)
        scores = scores.view([batch_size, 1, -1])
        C = (scores - self.anchors) ** 2
        C = C / (C.max().detach())
        f = torch.ones([batch_size, 1, self.n])
        g = torch.ones([batch_size, 2, 1])

        f = f.to(self.device)
        g = g.to(self.device)

        def min_epsilon_row(Z, epsilon):
            return -epsilon * torch.logsumexp((-C + f + g) / epsilon, -1, keepdim=True)

        def min_epsilon_col(Z, epsilon):
            return -epsilon * torch.logsumexp((-C + f + g) / epsilon, -2, keepdim=True)

        for i in range(self.max_iter):
            f = min_epsilon_col(C - f - g, self.epsilon) + f + self.epsilon * torch.log(self.mu)
            g = min_epsilon_row(C - f - g, self.epsilon) + g + self.epsilon * torch.log(self.nu)

        P = torch.exp((-C + f + g) / self.epsilon)

        return P


class TopK_custom(torch.nn.Module):
    def __init__(self, k, epsilon=0.1, max_iter=200, device: torch.device = torch.device('cpu')):
        super(TopK_custom, self).__init__()
        self.k = k
        self.epsilon = epsilon
        self.anchors = torch.FloatTensor([0, 1]).view([1, 1, 2])
        self.max_iter = max_iter
        self.device = device

        self.anchors = self.anchors.to(self.device)

    def forward(self, scores):
        bs, n = scores.size()
        scores = scores.view([bs, n, 1])

        # find the -inf value and replace it with the minimum value except -inf
        scores_ = scores.clone().detach()
        max_scores = torch.max(scores_).detach()
        scores_[scores_ == float('-inf')] = float('inf')
        min_scores = torch.min(scores_).detach()
        filled_value = min_scores - (max_scores - min_scores)
        mask = scores == float('-inf')
        scores.masked_fill(mask, filled_value)

        C = (scores - self.anchors) ** 2
        C = C / (C.max().detach())

        mu = torch.ones([1, n, 1], requires_grad=False) / n
        nu = torch.FloatTensor([self.k / n, (n - self.k) / n]).view([1, 1, 2])

        mu = mu.to(self.device)
        nu = nu.to(self.device)

        Gamma = TopKFunc.apply(C, mu, nu, self.epsilon, self.max_iter, self.device)

        return Gamma.transpose(-1, -2)


class TopKFunc(Function):
    @staticmethod
    def forward(ctx, C, mu, nu, epsilon, max_iter, device):
        bs, n, k_ = C.size()
        with torch.no_grad():
            f = torch.zeros([bs, n, 1]).to(device)
            g = torch.zeros([bs, 1, k_]).to(device)

            def min_epsilon_row(Z, epsilon):
                return -epsilon * torch.logsumexp((-C + f + g) / epsilon, -1, keepdim=True)

            def min_epsilon_col(Z, epsilon):
                return -epsilon * torch.logsumexp((-C + f + g) / epsilon, -2, keepdim=True)

            for i in range(max_iter):
                f = min_epsilon_row(C - f - g, epsilon) + f + epsilon * torch.log(mu)
                g = min_epsilon_col(C - f - g, epsilon) + g + epsilon * torch.log(nu)
            f = min_epsilon_row(C - f - g, epsilon) + f + epsilon * torch.log(mu)

            Gamma = torch.exp((-C + f + g) / epsilon)

            ctx.save_for_backward(mu, nu, Gamma)
            ctx.epsilon = epsilon
            ctx.device = device
        return Gamma

    @staticmethod
    def backward(ctx, grad_output_Gamma):

        epsilon = ctx.epsilon
        mu, nu, Gamma = ctx.saved_tensors
        # mu [1, n, 1]
        # nu [1, 1, k+1]
        # Gamma [bs, n, k+1]

        with torch.no_grad():
            nu_ = nu[:, :, :-1]
            Gamma_ = Gamma[:, :, :-1]

            bs, n, k_ = Gamma.size()

            inv_mu = 1. / (mu.view([1, -1]))  # [1, n]
            Kappa = torch.diag_embed(nu_.squeeze(-2)) \
                    - torch.matmul(Gamma_.transpose(-1, -2) * inv_mu.unsqueeze(-2),
                                   Gamma_)  # [bs, k, k]
            # print(Kappa, Gamma_)
            padding_value = 1e-10
            ridge = torch.ones([bs, k_ - 1]).diag_embed().to(ctx.device)
            inv_Kappa = torch.inverse(Kappa + ridge * padding_value)  # [bs, k, k]
            # print(Kappa, inv_Kappa)
            mu_Gamma_Kappa = (inv_mu.unsqueeze(-1) * Gamma_).matmul(inv_Kappa)  # [bs, n, k]
            H1 = inv_mu.diag_embed() + mu_Gamma_Kappa.matmul(
                Gamma_.transpose(-1, -2)) * inv_mu.unsqueeze(-2)  # [bs, n, n]
            H2 = - mu_Gamma_Kappa  # [bs, n, k]
            H3 = H2.transpose(-1, -2)  # [bs, k, n]
            H4 = inv_Kappa  # [bs, k, k]

            H2_pad = F.pad(H2, pad=(0, 1), mode='constant', value=0)
            H4_pad = F.pad(H4, pad=(0, 1), mode='constant', value=0)
            grad_f_C = H1.unsqueeze(-1) * Gamma.unsqueeze(-3) \
                       + H2_pad.unsqueeze(-2) * Gamma.unsqueeze(-3)  # [bs, n, n, k+1]
            grad_g_C = H3.unsqueeze(-1) * Gamma.unsqueeze(-3) \
                       + H4_pad.unsqueeze(-2) * Gamma.unsqueeze(-3)  # [bs, k, n, k+1]

            grad_g_C_pad = F.pad(grad_g_C, pad=(0, 0, 0, 0, 0, 1), mode='constant', value=0)
            grad_C1 = grad_output_Gamma * Gamma
            grad_C2 = torch.sum(grad_C1.view([bs, n, k_, 1, 1]) * grad_f_C.unsqueeze(-3),
                                dim=(1, 2))
            grad_C3 = torch.sum(grad_C1.view([bs, n, k_, 1, 1]) * grad_g_C_pad.unsqueeze(-4),
                                dim=(1, 2))

            grad_C = (-grad_C1 + grad_C2 + grad_C3) / epsilon

        return grad_C, None, None, None, None, None


if __name__ == '__main__':
    import numpy as np

    K = 6
    USE_MANUAL_GRAD = False
    EPSILON = 1e-5
    INNER_ITER = 500
    NUM_SAMPLES = 100
    NUM_TRAIN_NEIGHBORS = 20

    DATABASE_SIZE = 20
    EMBEDDING_SIZE = 50
    QUERY_SIZE = 5

    dknn = DKNN(
        k=K,
        num_samples=NUM_SAMPLES,
        num_neighbors=NUM_TRAIN_NEIGHBORS,
        use_manual_grad=USE_MANUAL_GRAD,
        epsilon=EPSILON,
        max_iter=INNER_ITER
    )

    neighbors = torch.normal(mean=0, std=1, size=(DATABASE_SIZE, EMBEDDING_SIZE))
    query = torch.normal(mean=0, std=1, size=(QUERY_SIZE, EMBEDDING_SIZE))

    top_k = dknn.forward(query=query, neighbors=neighbors, tau=1.0)

    print(top_k.size())
    print(top_k)
