"""
Transforms performed on angles l1, l1p, l2
"""

import numpy as np
import torch


# def fourier(N, k) -> np.ndarray: # TODO. Define an orthonormal Fourier basis.


class FourierAngle:
    """
    Perform a Fourier transform along angles l1, l1p, l2.
    """
    def __init__(self):
        self.F = None

    def __call__(self, s_cov, idx_info):
        cov_type, j1, a, b, l1, l2, l3 = idx_info.T

        L = l3.max() + 1  # number of angles # TODO. Hack, should better infer the value of L

        # computes Fourier along angles
        C01re = s_cov[:, cov_type == 'C01re'].reshape(-1, L)
        C01im = s_cov[:, cov_type == 'C01im'].reshape(-1, L)
        C11re = s_cov[:, cov_type == 'C11re'].reshape(-1, L, L)
        C11im = s_cov[:, cov_type == 'C11im'].reshape(-1, L, L)

        # idx_info for mean, P00, S1
        cov_no_fourier = s_cov[0, np.isin(cov_type, ['mean', 'P00', 'S1'])]
        idx_info_no_fourier = idx_info[np.isin(cov_type, ['mean', 'P00', 'S1']), :]

        # idx_info for C01
        C01_f_flattened = torch.cat([C01_f.real.reshape(-1), C01_f.imag.reshape(-1)])
        idx_info_C01 = idx_info[np.isin(cov_type, ['C01re', 'C01im']), :]

        # idx_info for C11
        C11_f_flattened = torch.cat([C11_f.real.reshape(-1), C11_f.imag.reshape(-1)])
        idx_info_C11 = idx_info[np.isin(cov_type, ['C11re', 'C11im']), :]

        idx_info_f = np.concatenate([idx_info_no_fourier, idx_info_C01, idx_info_C11])
        s_covs_f = torch.cat([cov_no_fourier, C01_f_flattened, C11_f_flattened])

        return s_covs_f, idx_info_f
