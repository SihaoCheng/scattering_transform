import numpy as np
import torch
import torch.fft

class ST_2d(object):
    def __init__(self, filters_set):
        self.filters_set = filters_set

    def forward(self, data, J, L, backend='torch',
                j1j2_criteria='j2>j1', mask=None, pseudo_coef=1):
        """
        calculate the scattering coefficients of a 2D image.
        data: array, with the same size as the filters in self.filters_set
        J:    int, number of wavelet scales used
        j1j2_criteria: str, an expression with j1 and j2
        mask: None or 2D-array, indicating the weight of pixels when taking 
            the spatial average in S_n = <I_n>
        pseudo_coef: the power after taking modulus.
        """
        
        filters_set = self.filters_set
        if mask is not None:
            mask /= mask.mean() # normalize the mask array
        else:
            mask = 1

        S_0 = np.zeros(1, dtype=data.dtype)
        S_1 = np.zeros((J,L), dtype=data.dtype)
        S_2 = np.zeros((J,L,J,L), dtype=data.dtype)
        S_2_reduced = np.zeros((J,J,L), dtype=data.dtype)
        
        S_0[0] = (data * mask).mean()
        
        if backend=='torch':
            data = torch.from_numpy(data)
            data_f = torch.fft.fftn(data, dim=(-2,-1))
            
            # calculate 1st-order coefficients
            for j1 in np.arange(J):
                for l1 in np.arange(L):
                    I_1_temp  = torch.fft.ifftn(
                        data_f * filters_set['psi'][j1*L+l1][0],
                        dim=(-2,-1),
                    ).abs()**pseudo_coef
                    S_1[j1,l1] = (I_1_temp.numpy() * mask).mean()
                    
                    # calculate 2nd-order coefficients
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
                        data_f * filters_set['psi'][j1*L+l1][0]# filters should be np.array
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
                                
        for l1 in range(L):
            for l2 in range(L):
                S_2_reduced[:,:,(l2-l1)%L] += S_2[:,l1,:,l2]
        S_2_reduced /= L

        if input_dtype=='numpy':
            S = np.concatenate(( S_0, S_1.sum(1), S_2_reduced.flatten() ))
        if input_dtype=='torch':
            S = torch.cat(( S_0, S_1.sum(1), S_2_reduced.flatten()  )).numpy()
        return S, S_0, S_1, S_2

class PH_2d(object):
    def __init__(self, M, N, J, L, filters_set):
        self.M = M
        self.N = N
        self.J = J
        self.L = L
        self.filters = torch.zeros((J, L, M, N), torch.float32)
        for j in np.arange(J):
            for l in np.arange(L):
                self.filters[j, l, :, :] = filters_set['psi'][j*L+l][0]
    
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



class FiltersSet(object):
    def __init__(self, M, N, J, L):
        self.M = M
        self.N = N
        self.J = J
        self.L = L

    def generate_morlet(self, if_save=False, save_dir=None, precision='single'):
        psi = []
        for j in range(self.J):
            for theta in range(self.L):
                wavelet = self.morlet_2d(
                    M=self.M,
                    N=self.N,
                    sigma=0.8 * 2**j,
                    theta=(int(self.L-self.L/2-1)-theta) * np.pi / self.L,
                    xi=3.0 / 4.0 * np.pi /2**j,
                    slant=4.0/self.L,
                )
                wavelet_Fourier = np.fft.fft2(wavelet)
                wavelet_Fourier[0,0] = 0
                if precision=='double':
                    psi.append([torch.from_numpy(wavelet_Fourier.real)])
                if precision=='single':
                    psi.append([torch.from_numpy(wavelet_Fourier.real.astype(np.float32))])

        filters_set_mycode = {'psi':psi}
        if if_save:
            np.save(
                save_dir + 'filters_set_mycode_M' + str(self.M) + 'N' + str(self.N)
                + 'J' + str(self.J) + 'L' + str(self.L) + '_' + precision + '.npy',
                np.array([{'filters_set': filters_set_mycode}])
            )
        else:
            return filters_set_mycode

    def morlet_2d(self, M, N, sigma, theta, xi, slant=0.5, offset=0, fft_shift=False):
        """
            Computes a 2D Morlet filter.
            A Morlet filter is the sum of a Gabor filter and a low-pass filter
            to ensure that the sum has exactly zero mean in the temporal domain.
            It is defined by the following formula in space:
            psi(u) = g_{sigma}(u) (e^(i xi^T u) - beta)
            where g_{sigma} is a Gaussian envelope, xi is a frequency and beta is
            the cancelling parameter.

            Parameters
            ----------
            M, N : int
                spatial sizes
            sigma : float
                bandwidth parameter
            xi : float
                central frequency (in [0, 1])
            theta : float
                angle in [0, pi]
            slant : float, optional
                parameter which guides the elipsoidal shape of the morlet
            offset : int, optional
                offset by which the signal starts
            fft_shift : boolean
                if true, shift the signal in a numpy style

            Returns
            -------
            morlet_fft : ndarray
                numpy array of size (M, N)
        """
        wv = self.gabor_2d(M, N, sigma, theta, xi, slant, offset, fft_shift)
        wv_modulus = self.gabor_2d(M, N, sigma, theta, 0, slant, offset, fft_shift)
        K = np.sum(wv) / np.sum(wv_modulus)

        mor = wv - K * wv_modulus
        return mor

    def gabor_2d(self, M, N, sigma, theta, xi, slant=1.0, offset=0, fft_shift=False):
        """
            Computes a 2D Gabor filter.
            A Gabor filter is defined by the following formula in space:
            psi(u) = g_{sigma}(u) e^(i xi^T u)
            where g_{sigma} is a Gaussian envelope and xi is a frequency.

            Parameters
            ----------
            M, N : int
                spatial sizes
            sigma : float
                bandwidth parameter
            xi : float
                central frequency (in [0, 1])
            theta : float
                angle in [0, pi]
            slant : float, optional
                parameter which guides the elipsoidal shape of the morlet
            offset : int, optional
                offset by which the signal starts
            fft_shift : boolean
                if true, shift the signal in a numpy style

            Returns
            -------
            morlet_fft : ndarray
                numpy array of size (M, N)
        """
        gab = np.zeros((M, N), np.complex128)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float64)
        R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float64)
        D = np.array([[1, 0], [0, slant * slant]])
        curv = np.dot(R, np.dot(D, R_inv)) / ( 2 * sigma * sigma)

        for ex in [-2, -1, 0, 1, 2]:
            for ey in [-2, -1, 0, 1, 2]:
                [xx, yy] = np.mgrid[offset + ex * M:offset + M + ex * M, offset + ey * N:offset + N + ey * N]
                arg = -(curv[0, 0] * np.multiply(xx, xx) + (curv[0, 1] + curv[1, 0]) * np.multiply(xx, yy) + curv[
                    1, 1] * np.multiply(yy, yy)) + 1.j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
                gab = gab + np.exp(arg)

        norm_factor = (2 * np.pi * sigma * sigma / slant)
        gab = gab / norm_factor

        if (fft_shift):
            gab = np.fft.fftshift(gab, axes=(0, 1))
        return gab
