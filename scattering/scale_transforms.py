"""
Transforms performed on scales j1, j1p, j2
"""

import itertools

import numpy as np
import torch


def cosinus1(N):
    """
    The cosine basis cos(kpi (. + 0.5) / N) for 0 <= k < N.

    :param N:
    :return:
    """
    if N == 0:
        return np.zeros((0, 0))

    ts = np.linspace(0, np.pi * (N - 1) / N, N) + (0.5 * np.pi / N)
    indices = np.stack([k * ts for k in range(N)], axis=0)

    F = np.cos(indices)
    F[1:, :] *= np.sqrt(2 / N)
    F[0, :] *= np.sqrt(1 / N)

    return F


def init_cosine_projector(s_cov, idx_info):
    s_cov_flattened = s_cov
    cov_type, j1_s, a_s, b_s, l1_s, l1p_s, l2_s = idx_info.T
    idx_info_no_j1_l = np.stack([cov_type, l1_s, l1p_s, l2_s, a_s, b_s]).T.tolist()
    
    idx_info_no_j1_l = list(set([tuple(row) for row in idx_info_no_j1_l]))
    
#     idx_info_no_j1_l = np.array(idx_info_no_j1_l)
#     _, idx = np.unique(idx_info_no_j1_l, axis=0, return_index=True)
#     idx_info_no_j1_l = idx_info_no_j1_l[np.sort(idx)]

    J = a_s.max() + 1  # TODO. Hack, should infer better the value of J

    proj_l = []
    idx_info_out_l = []
    for (c_type, l1, l1p, l2, a, b) in idx_info_no_j1_l:
        idx = (cov_type == c_type) & (l1_s == l1) & (l1p_s == l1p) & (l2_s == l2) & (a_s == a) & (b_s == b)

        if c_type == 'mean':
            nb_om = 1
            proj = s_cov_flattened.new_zeros(nb_om, idx_info.shape[0])
            proj[0, idx] = 1.0
        if c_type in ['P00', 'S1']:
            nb_om = J
            proj = s_cov_flattened.new_zeros(nb_om, idx_info.shape[0])
            proj[:, idx] = torch.eye(J, dtype=s_cov_flattened.dtype, device=s_cov_flattened.device)
        if 'C01' in c_type:
            nb_om = J - a
            proj = s_cov_flattened.new_zeros(nb_om, idx_info.shape[0])
            proj[:, idx] = torch.tensor(cosinus1(nb_om), dtype=s_cov_flattened.dtype, device=s_cov_flattened.device)
        if 'C11' in c_type:
            nb_om = J + b - a
            proj = s_cov_flattened.new_zeros(nb_om, idx_info.shape[0])
            proj[:, idx] = torch.tensor(cosinus1(nb_om), dtype=s_cov_flattened.dtype, device=s_cov_flattened.device)

        if idx.sum() != nb_om:
            print("ERROR")

        idx_info_out_l.append([(c_type, om, a, b, l1, l1p, l2) for om in range(nb_om)])
        proj_l.append(proj)

    idx_info_out_l = list(itertools.chain.from_iterable(idx_info_out_l))
    P_cos = torch.cat(proj_l)

    return P_cos, idx_info_out_l


class FourierScale:
    """
    Perform a cosine transform along angles l1, l1p, l2.
    """
    def __init__(self):
        self.P_cos = None
        self.idx_info_out = None

    def __call__(self, s_cov, idx_info):
        # construct cosine projector P_cos that maps s_cov to s_cov_fourier
        if self.P_cos is None:
            self.P_cos, self.idx_info_out = init_cosine_projector(s_cov, idx_info)

        # compute cosine transform along j1
#         s_cov_f = self.P_cos @ s_cov
#         print(type(self.P_cos), type(s_cov))
        s_cov_f = (self.P_cos[None,...] @ s_cov[...,None])[...,0]

        return s_cov_f, self.idx_info_out
