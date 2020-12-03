import numpy as np
import torch
import torch.fft

class PH_mycode(object):
    def __init__(self, M, N, J, L, filters_set):
        self.M = M
        self.N = N
        self.J = J
        self.L = L
        self.filters = torch.zeros((J, L, M, N, 2))
        for j in np.arange(J):
            for l in np.arange(L):
                self.filters[j, l, :, :, 0] = filters_set['psi'][j*L+l][0]
    
    def phase_harmonic(self, data_modulus, data_phase, p):
        result = torch.cat(
            ((torch.cos(data_phase * p) * data_modulus)[...,None],
             (torch.sin(data_phase * p) * data_modulus)[...,None],
            ), -1
        )
        return result
    
    def forward(self, data, dj, dl,):

        C_00 = np.zeros((J, L, dj+1, dl+1), dtype=data.dtype)
        C_01 = np.zeros((J, L, dj+1, dl+1), dtype=data.dtype)
        C_ph = np.zeros((J, L, dj+1, dl+1), dtype=data.dtype)

        data = torch.from_numpy(np.concatenate((data[...,None], np.zeros_like(data[...,None])),-1))
        data_f = torch.fft(data, 2)
        I_conv_psi = torch.ifft(data_f[None,None,:,:,:] * self.filters, 2)
        I_conv_psi_modulus = (I_conv_psi**2).sum(-1)**0.5
        I_conv_psi_phase = torch.atan2(
            I_conv_psi[...,1], I_conv_psi[...,0]
        )
        
        for j in np.arange(J):
            for l in np.arange(L):
                for delta_j in np.arange(dj+1):
                    for delta_l in np.arange(dl+1):
                        if j+delta_j < J:# and not (delta_j==0 and delta_l!=0):
                            
                            C_01[j, l, delta_j, delta_l] = (
                                I_conv_psi_modulus[j,l,:,:] * \
                                I_conv_psi[j+delta_j,(l+delta_l)%L,:,:,0]
                            ).mean()
                            
                            if True:#delta_j > 0: #not to calculate PS
                                
                                C_00[j, l, delta_j, delta_l] = (
                                    I_conv_psi_modulus[j,l] * I_conv_psi_modulus[j+delta_j,(l+delta_l)%L]
                                ).mean() - \
                                I_conv_psi_modulus[j,l].mean() * I_conv_psi_modulus[j+delta_j,(l+delta_l)%L].mean()
                                
                                ph = self.phase_harmonic(
                                        I_conv_psi_modulus[j+delta_j,(l+delta_l)%L],
                                        I_conv_psi_phase  [j+delta_j,(l+delta_l)%L],
                                        2**delta_j
                                    )
                                C_ph[j, l, delta_j, delta_l] = (
                                    I_conv_psi[j,l,...,0] * ph[...,0] +
                                    I_conv_psi[j,l,...,1] * ph[...,1]
                                ).mean()
        return C_00.mean(1), C_01.mean(1), C_ph.mean(1)
