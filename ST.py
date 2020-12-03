import numpy as np
import torch
import torch.fft

class ST_mycode(object):
    def __init__(self, filters_set):
        self.filters_set = filters_set

    def forward(self, data, J, L, backend='torch', input_dtype='numpy',
                  j1j2_criteria='j2>j1', mask=None, pseudo_coef=1):
        # Number of coefficients per layer
        filters_set = self.filters_set
        self.M = data.shape[-2]
        self.N = data.shape[-1]

        if mask is not None:
            mask /= mask.mean()
        else:
            mask = 1

        if input_dtype=='numpy':
            S_0 = np.zeros(1, dtype=data.dtype)
            S_1 = np.zeros((J,L), dtype=data.dtype)
            S_2 = np.zeros((J,L,J,L), dtype=data.dtype)
            S_2_reduced = np.zeros((J,J,L), dtype=data.dtype)
        elif input_dtype=='torch':
            S_0 = torch.zeros(1, dtype=data.dtype)
            S_1 = torch.zeros((J,L), dtype=data.dtype)
            S_2 = torch.zeros((J,L,J,L), dtype=data.dtype)
            S_2_reduced = torch.zeros((J,J,L), dtype=data.dtype)
        S_0[0] = data.mean()
        
        if backend=='torch':
            if input_dtype=='numpy':
                data = torch.from_numpy(data)
            data_f = torch.fft.fftn(data, dim=(-2,-1))
            for j1 in np.arange(J):
                for l1 in np.arange(L):
                    I_1_temp  = torch.fft.ifftn(
                        data_f * filters_set['psi'][j1*L+l1][0],
                        dim=(-2,-1),
                    ).abs()**pseudo_coef
                    S_1[j1,l1] = (I_1_temp.numpy() * mask).mean()

                    I_1_temp_f = torch.fft.fftn(I_1_temp, dim=(-2,-1))
                    for j2 in np.arange(J):
                        if eval(j1j2_criteria):
                            for l2 in np.arange(L):
                                I_2_temp = torch.fft.ifftn(
                                    I_1_temp_f * filters_set['psi'][j2*L+l2][0],
                                    dim=(-2,-1),
                                ).abs()**pseudo_coef
                                S_2[j1,l1,j2,l2] = (I_2_temp.numpy() * mask).mean()

        if backend=='numpy':
            data_f = np.fft.fft2(data)
            for j1 in np.arange(J):
                for l1 in np.arange(L):
                    I_1_temp  = np.abs(np.fft.ifft2(
                         data_f * filters_set['psi'][j1*L+l1][0]
                    ))**pseudo_coef
                    S_1[j1,l1] = (I_1_temp * mask).mean()

                    I_1_temp_f = np.fft.fft2(I_1_temp)
                    for j2 in np.arange(J):
                        if eval(j1j2_criteria):
                            for l2 in np.arange(L):
                                I_2_temp = np.abs(np.fft.ifft2(
                                    I_1_temp_f * filters_set['psi'][j2*L+l2][0]
                                ))**pseudo_coef
                                S_2[j1,l1,j2,l2] = (I_2_temp * mask).mean()
                                # what about masks?
                                
        for l1 in range(L):
            for l2 in range(L):
                S_2_reduced[:,:,(l2-l1)%L] += S_2[:,l1,:,l2]
        S_2_reduced /= L

        if input_dtype=='numpy':
            S = np.concatenate(( S_0, S_1.sum(1), S_2_reduced.flatten() ))
        if input_dtype=='torch':
            S = torch.cat(( S_0, S_1.sum(1), S_2_reduced.flatten()  )).numpy()
        return S, S_0, S_1, S_2
