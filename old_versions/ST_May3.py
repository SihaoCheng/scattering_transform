# Updates:
# May 3, 2021: 
# 1. Updated the 'fast algorithm', which now can be used on images
#    with arbitary sizes. The fractional difference between the 
#    coefficients calculated using 'classic' and 'fast' algorithm 
#    is in general less than 1/1000.
# 2. Important! Now the weight function is fed when initializing an
#    ST_2D object (which is a calculator), not when forward-running it.
# 3. Added a function for image pre-processing (removing the overall
#    slope, in order to reduce edge effect).
# 4. Added a function for ST reduction. (not mature yet)
#
#
#
# Mar 19, 2021: 
# Several optimations from kymatio:
# 1. Updated to PyTorch 1.7.
# 2. Added a 'fast algorithm' with almost unchanged results, reducing
#    calculation time by 5x or more. Unfortunately it only works for
#    dyadic image sizes now, such as 256 * 128.
# 3. Rewrote the function to generate wavelet filters, which is much faster now.
#
#
#
# Author: Sihao Cheng, Johns Hopkins University
# If you have any questions or suggestions, please do not hesitate
# to contact me: s.cheng@jhu.edu
#

import numpy as np
import torch
import torch.fft

class ST_2D(object):
    def __init__(self, filters_set, J, L, device='cpu', weight=None):
        self.M, self.N = filters_set['psi'][0][0].shape
        dtype = filters_set['psi'][0][0].dtype
        # filters set
        self.filters_set = torch.zeros((J,L,self.M,self.N), dtype=dtype)
        if len(filters_set['psi'][0]) == 1:
            for j in range(J):
                for l in range(L):
                    self.filters_set[j,l] = filters_set['psi'][j*L+l][0]
        else:
            self.filters_set = filters_set['psi']
        # weight
        if weight is None:
            self.weight = None
            self.weight_f = None
        else:
            if self.M!=weight.shape[0] or self.N!=weight.shape[1]:
                print('"weight" must have the same image size as filters in "filters_set".')
            self.weight = torch.from_numpy(weight / weight.mean())
            self.weight_f = torch.fft.fftn(self.weight, dim=(-2,-1))
            self.weight_downsample_list = []
            for j in np.arange(J):
                dx, dy = self.get_dxdy(j)
                weight_downsample = torch.fft.ifftn(
                    self.cut_high_k_off(self.weight_f, dx, dy),
                    dim=(-2,-1)
                ).real
                if device=='gpu':
                    weight_downsample = weight_downsample.cuda()
                self.weight_downsample_list.append(
                    weight_downsample / weight_downsample.mean()
                )
        # device
        self.device = device
        if device=='gpu':
            self.filters_set = self.filters_set.cuda()
            if weight is not None:
                self.weight = self.weight.cuda()

    def cut_high_k_off(self, data_f, dx, dy):
        if_xodd = (self.M%2==1)
        if_yodd = (self.N%2==1)
        result = torch.cat(
            (torch.cat(
                ( data_f[...,:dx+if_xodd, :dy+if_yodd] , data_f[...,-dx:, :dy+if_yodd]
                ), -2),
              torch.cat(
                ( data_f[...,:dx+if_xodd, -dy:] , data_f[...,-dx:, -dy:]
                ), -2)
            ),-1)
        return result

    def get_dxdy(self, j):
        dx = int(max( 16, min( np.ceil(self.M/2**j), self.M//2 ) ))
        dy = int(max( 16, min( np.ceil(self.N/2**j), self.N//2 ) ))
        return dx, dy

    def forward(self, data, J, L, algorithm='classic',
                j1j2_criteria='j2>j1', pseudo_coef=1,
                ):
        '''
            Calculates the scattering coefficients for a set of images.
            Parameters
            ----------
            data : numpy array or torch tensor
                image set, with size [N_image, x-sidelength, y-sidelength]
            J, L : int
                the number of scales and angles for calculation
            algorithm: 'classic' or 'fast', optional
                'classic' uses the full Fourier space to calculate every
                scattering coefficients.
                'fast' uses only the inner part of the Fourier space to 
                calculate the large-scale scattering coefficients.
            j1j2_criteria : str, optional
                which S2 coefficients to calculate. Default is 'j2>j1'. 
            pseudo_coef : float/int, optional
                the power of modulus. This allows for a modification of 
                the scattering transform. For standard ST, it should be 
                one, which is also the default. When it is 2, the n-th
                order ST will become some 2^n point functions.
            Returns
            -------
            S : torch tensor
                reduced ST coef (averaged over angles), flattened, with size
                [N_image, 1 + J + J*J*L]
            S0 : torch tensor
                0th order ST coefficients, with size [N_image, 1]
            S1 : torch tensor
                1st order ST coefficients, with size [N_image, J, L]
            S2 : torch tensor
                2nd order ST coefficients, with size [N_image, J, L, J, L]
            E : torch tensor
                power in each 1st-order wavelet bands, with size 
                [N_image, J, L]
            E_residual: torch tensor
                residual power in the 2nd-order scattering fields, which is 
                the residual power not extracted by the scattering coefficients.
                it has a size of [N_image, J, L, J, L].

        '''
        M, N = self.M, self.N
        N_image = data.shape[0]
        filters_set = self.filters_set
        weight = self.weight

        # convert numpy array input into torch tensors
        if type(data) == np.ndarray:
            data = torch.from_numpy(data)

        # initialize tensors for scattering coefficients
        S0 = torch.zeros((N_image,1), dtype=data.dtype)  
        S1 = torch.zeros((N_image,J,L), dtype=data.dtype)
        S2 = torch.zeros((N_image,J,L,J,L), dtype=data.dtype)
        S2_reduced = torch.zeros((N_image,J,J,L), dtype=data.dtype)
        E = torch.zeros((N_image,J,L), dtype=data.dtype)
        E_residual = torch.zeros((N_image,J,L,J,L), dtype=data.dtype)
        
        # move torch tensors to gpu device, if required
        if self.device=='gpu':
            data = data.cuda()
            S0 = S0.cuda()
            S1 = S1.cuda()
            S2 = S2.cuda()
            S2_reduced = S2_reduced.cuda()
            E = E.cuda()
            E_residual = E_residual.cuda()
        
        # 0th order
        S0[:,0] = data.mean((-2,-1))
        
        # 1st and 2nd order
        data_f = torch.fft.fftn(data, dim=(-2,-1))
        
        if algorithm == 'classic':
            # calculating scattering coefficients, with two Fourier transforms
            if weight is None:
                weight_temp = 1
            else:
                weight_temp = weight[None,None,None,:,:]
            # 1st-order scattering field
            I1 = torch.fft.ifftn(
                data_f[:,None,None,:,:] * filters_set[None,:J,:,:,:],
                dim=(-2,-1),
            ).abs()**pseudo_coef
            # coefficients
            S1 = (I1 * weight_temp).mean((-2,-1))
            E = (I1**2 * weight_temp).mean((-2,-1))

            # 2nd order
            I1_f = torch.fft.fftn(I1, dim=(-2,-1))
            for j1 in np.arange(J):
                for j2 in np.arange(J):
                    if eval(j1j2_criteria):
                        # scattering field
                        I2_temp = torch.fft.ifftn(
                            I1_f[:,j1,:,None,:,:] * filters_set[None,j2,None,:,:,:], 
                            dim=(-2,-1),
                        ).abs()**pseudo_coef
                        # coefficients
                        S2[:,j1,:,j2,:] = (I2_temp * weight_temp).mean((-2,-1))
                        E_residual[:,j1,:,j2,:] = (
                            (I2_temp - I2_temp.mean((-2,-1))[:,:,:,None,None])**2 * 
                            weight_temp
                        ).mean((-2,-1))

        if algorithm == 'fast':
            # only use the low-k Fourier coefs when calculating large-j scattering coefs.
            for j1 in np.arange(J):
                # 1st order: cut high k
                dx1, dy1 = self.get_dxdy(j1)
                data_f_small = self.cut_high_k_off(data_f, dx1, dy1)
                wavelet_f = self.cut_high_k_off(filters_set[j1], dx1, dy1)
                _, M1, N1 = wavelet_f.shape
                # scattering field
                I1_temp  = torch.fft.ifftn(
                    data_f_small[:,None,:,:] * wavelet_f[None,:,:,:],
                    dim=(-2,-1),
                ).abs()**pseudo_coef
                # coefficients
                if weight is None:
                    weight_temp = 1
                else:
                    weight_temp = self.weight_downsample_list[j1][None,None,:,:]
                S1[:,j1] = (I1_temp * weight_temp).mean((-2,-1)) * M1*N1/M/N
                E[:,j1] = (I1_temp**2 * weight_temp).mean((-2,-1)) * (M1*N1/M/N)**2
                
                # 2nd order
                I1_temp_f = torch.fft.fftn(I1_temp, dim=(-2,-1))
                for j2 in np.arange(J):
                    if eval(j1j2_criteria):
                        # cut high k
                        dx2, dy2 = self.get_dxdy(j2)
                        I1_temp_f_small = self.cut_high_k_off(I1_temp_f, dx2, dy2)
                        wavelet_f2 = self.cut_high_k_off(filters_set[j2], dx2, dy2)
                        _, M2, N2 = wavelet_f2.shape
                        # scattering field
                        I2_temp = torch.fft.ifftn(
                            I1_temp_f_small[:,:,None,:,:] * wavelet_f2[None,None,:,:,:], 
                            dim=(-2,-1),
                        ).abs()**pseudo_coef
                        # coefficients
                        if weight is None:
                            weight_temp = 1
                        else:
                            weight_temp = self.weight_downsample_list[j2][None,None,None,:,:]
                        S2[:,j1,:,j2,:] = (I2_temp * weight_temp).mean((-2,-1)) * M2*N2/M/N
                        E_residual[:,j1,:,j2,:] = (
                            (I2_temp - I2_temp.mean((-2,-1))[:,:,:,None,None])**2 *
                            weight_temp
                        ).mean((-2,-1)) * (M2*N2/M/N)**2
        # average over l1
        for l1 in range(L):
            for l2 in range(L):
                S2_reduced[:,:,:,(l2-l1)%L] += S2[:,:,l1,:,l2]
        S2_reduced /= L

        S = torch.cat(( S0, S1.sum(-1), S2_reduced.reshape((N_image,-1))  ), 1)
        # return S.cpu().numpy(), S0.cpu().numpy(), S1.cpu().numpy(), S2.cpu().numpy(),\
        #        E.cpu().numpy(), E_residual.cpu().numpy()
        return S, S0, S1, S2, E, E_residual


class FiltersSet(object):
    def __init__(self, M, N, J, L):
        self.M = M
        self.N = N
        self.J = J
        self.L = L

    def generate_morlet(self, if_save=False, save_dir=None, precision='double'):
        if precision=='double':
            psi = torch.zeros((self.J, self.L, self.M, self.N), dtype=torch.float64)
        if precision=='single':
            psi = torch.zeros((self.J, self.L, self.M, self.N), dtype=torch.float32)
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
                    psi[j, theta] = torch.from_numpy(wavelet_Fourier.real)
                if precision=='single':
                    psi[j, theta] = torch.from_numpy(wavelet_Fourier.real.astype(np.float32))
        if precision=='double':
            phi = torch.from_numpy(
                self.gabor_2d_mycode(self.M, self.N, 0.8 * 2**(self.J-1), 0, 0).real
            )
        if precision=='single':
            phi = torch.from_numpy(
                self.gabor_2d_mycode(self.M, self.N, 0.8 * 2**(self.J-1), 0, 0).real.astype(np.float32)
            )
        
        filters_set = {'psi':psi, 'phi':phi}
        if if_save:
            np.save(
                save_dir + 'filters_set_mycode_M' + str(self.M) + 'N' + str(self.N)
                + 'J' + str(self.J) + 'L' + str(self.L) + '_' + precision + '.npy', 
                np.array([{'filters_set': filters_set}])
            )
        return filters_set

    def morlet_2d(self, M, N, sigma, theta, xi, slant=0.5, offset=0, fft_shift=False):
        """
            (from kymatio package) 
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
        wv = self.gabor_2d_mycode(M, N, sigma, theta, xi, slant, offset, fft_shift)
        wv_modulus = self.gabor_2d_mycode(M, N, sigma, theta, 0, slant, offset, fft_shift)
        K = np.sum(wv) / np.sum(wv_modulus)

        mor = wv - K * wv_modulus
        return mor

    def gabor_2d_mycode(self, M, N, sigma, theta, xi, slant=1.0, offset=0, fft_shift=False):
        """
            (partly from kymatio package)
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
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float64)
        R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float64)
        D = np.array([[1, 0], [0, slant * slant]])
        curv = np.matmul(R, np.matmul(D, R_inv)) / ( 2 * sigma * sigma)

        gab = np.zeros((M, N), np.complex128)
        xx = np.empty((2,2, M, N))
        yy = np.empty((2,2, M, N))

        for ii, ex in enumerate([-1, 0]):
            for jj, ey in enumerate([-1, 0]):
                xx[ii,jj], yy[ii,jj] = np.mgrid[
                    offset + ex * M : offset + M + ex * M, 
                    offset + ey * N : offset + N + ey * N
                ]
        
        arg = -(curv[0, 0] * xx * xx + (curv[0, 1] + curv[1, 0]) * xx * yy + curv[1, 1] * yy * yy) +\
            1.j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
        gab = np.exp(arg).sum((0,1))

        norm_factor = 2 * np.pi * sigma * sigma / slant
        gab = gab / norm_factor

        if fft_shift:
            gab = np.fft.fftshift(gab, axes=(0, 1))
        return gab
        

def remove_slope(images):
    M = images.shape[-2]
    N = images.shape[-1]
    z = images
    x = np.arange(M)[None,:,None]
    y = np.arange(N)[None,None,:]
    k_x = (
        (x - x.mean(-2)[:,None,:]) * (z - z.mean(-2)[:,None,:])
    ).mean((-2,-1)) / ((x - x.mean(-2)[:,None,:])**2).mean((-2,-1))
    k_y = (
        (y - y.mean(-1)[:,:,None]) * (z - z.mean(-1)[:,:,None])
    ).mean((-2,-1)) / ((y - y.mean(-1)[:,:,None])**2).mean((-2,-1))

    return z - k_x[:,None,None] * (x-M//2) - k_y[:,None,None] * (y-N//2)


def reduced_ST(S, J, L):
    s0 = S[:,0:1]
    s1 = S[:,1:J+1]
    s2 = S[:,J+1:].reshape((-1,J,J,L))
    s21 = (s2.mean(-1) / s1[:,:,None]).reshape((-1,J**2))
    s22 = (s2[:,:,:,0] / s2[:,:,:,L//2]).reshape((-1,J**2))
    
    s1 = np.log(s1)
    select = s21[0]>0
    s21 = np.log(s21[:, select])
    s22 = np.log(s22[:, select])
    
    j1 = (np.arange(J)[:,None] + np.zeros(J)[None,:]).flatten()
    j2 = (np.arange(J)[None,:] + np.zeros(J)[:,None]).flatten()
    j1j2 = np.concatenate((j1[None, select], j2[None, select]), axis=0)
    return s0, s1, s21, s22, s2, j1j2


class Bispectrum_Calculator(object):
    def __init__(self, k_range, M, N, device='cpu'):
        self.device = device
        self.k_range = k_range
        self.M = M
        self.N = N
        X, Y = np.meshgrid(np.arange(M), np.arange(N))
        d = ((X-M//2)**2+(Y-N//2)**2)**0.5
        
        self.k_filters = np.zeros((len(k_range)-1, M, N), dtype=bool)
        for i in range(len(k_range)-1):
            self.k_filters[i,:,:] = np.fft.ifftshift((d<=k_range[i+1]) * (d>k_range[i]))
        self.k_filters_torch = torch.from_numpy(self.k_filters)
        refs = torch.fft.ifftn(self.k_filters_torch, dim=(-2,-1)).real
        
        self.select = torch.zeros(
            (len(self.k_range)-1, len(self.k_range)-1, len(self.k_range)-1), 
            dtype=bool
        )
        self.B_ref_array = torch.zeros(
            (len(self.k_range)-1, len(self.k_range)-1, len(self.k_range)-1),
            dtype=torch.float32
        )
        for i1 in range(len(self.k_range)-1):
            for i2 in range(i1+1):
                for i3 in range(i2+1):
                    if i2 + i3 >= i1 :
                        self.select[i1, i2, i3] = True
                        self.B_ref_array[i1, i2, i3] = (refs[i1] * refs[i2] * refs[i3]).mean()
        if device=='gpu':
            self.k_filters_torch = self.k_filters_torch.cuda()
            self.select = self.select.cuda()
            self.B_ref_array = self.B_ref_array.cuda()

        
    def forward(self, image):
        if type(image) == np.ndarray:
            image = torch.from_numpy(image)

        B_array = torch.zeros(
            (len(self.k_range)-1, len(self.k_range)-1, len(self.k_range)-1), 
            dtype=image.dtype
        )
        if self.device=='gpu':
            B_array = B_array.cuda()
        image_f = torch.fft.fftn(image, dim=(-2,-1))
        convs = torch.fft.ifftn(
            image_f[None,...] * self.k_filters_torch,
            dim=(-2,-1)
        ).real
        convs_std = convs.std((-1,-2))
        for i1 in range(len(self.k_range)-1):
            for i2 in range(i1+1):
                for i3 in range(i2+1):
                    if i2 + i3 >= i1 :
                        B = convs[i1] * convs[i2] * convs[i3]
                        B_array[i1, i2, i3] = B.mean() *1e8 # / self.B_ref_array[i1, i2, i3]
        return B_array[self.select]
        
        
def get_power_spectrum(target, bins, device='cpu'):
    '''
    get the power spectrum of a given image
    '''
    M, N = target.shape
    modulus = torch.fft.fftn(target, dim=(-2,-1)).abs()
    
    modulus = torch.cat(
        ( torch.cat(( modulus[M//2:, N//2:], modulus[:M//2, N//2:] ), 0),
          torch.cat(( modulus[M//2:, :N//2], modulus[:M//2, :N//2] ), 0)
        ),1)
    X = torch.arange(0,M)
    Y = torch.arange(0,N)
    Ygrid, Xgrid = torch.meshgrid(Y,X)
    R = ((Xgrid - M/2)**2 + (Ygrid - N/2)**2)**0.5

    R_range = torch.logspace(0.0, np.log10(1.4*M/2), bins)
    R_range = torch.cat((torch.tensor([0]), R_range))
    power_spectrum = torch.zeros(len(R_range)-1, dtype=target.dtype)
    if device=='gpu':
        R = R.cuda()
        R_range = R_range.cuda()
        power_spectrum = power_spectrum.cuda()

    for i in range(len(R_range)-1):
        select = (R >= R_range[i]) * (R < R_range[i+1])
        power_spectrum[i] = modulus[select].mean()
    return power_spectrum, R_range
    
    
def get_random_data(target, M, N, mode='image'):
    '''
    get a gaussian random field with the same power spectrum as the image 'target' (in the 'image' mode),
    or with an assigned power spectrum function 'target' (in the 'func' mode).
    '''
    
    if mode == 'func':
        random_phase = np.random.normal(0,1,(M//2-1,N-1)) + np.random.normal(0,1,(M//2-1,N-1))*1j
        random_phase_left = (np.random.normal(0,1,(M//2-1)) + np.random.normal(0,1,(M//2-1))*1j)[:,None]
        random_phase_top = (np.random.normal(0,1,(N//2-1)) + np.random.normal(0,1,(N//2-1))*1j)[None,:]
        random_phase_middle = (np.random.normal(0,1,(N//2-1)) + np.random.normal(0,1,(N//2-1))*1j)[None,:]
        random_phase_corners = np.random.normal(0,1,3)
    if mode == 'image':
        random_phase = np.random.rand(M//2-1,N-1)
        random_phase_left = np.random.rand(M//2-1)[:,None]
        random_phase_top = np.random.rand(N//2-1)[None,:]
        random_phase_middle = np.random.rand(N//2-1)[None,:]
        random_phase_corners = np.random.randint(0,2,3)/2
    gaussian_phase = np.concatenate((
                      np.concatenate((random_phase_corners[1][None,None],
                                      random_phase_left,
                                      random_phase_corners[2][None,None],
                                      -random_phase_left[::-1,:],
                                    ),axis=0),
                      np.concatenate((np.concatenate((random_phase_top,
                                                      random_phase_corners[0][None,None],
                                                      -random_phase_top[:,::-1],
                                                    ),axis=1),
                                      random_phase, 
                                      np.concatenate((random_phase_middle, 
                                                      np.array(0)[None,None], 
                                                      -random_phase_middle[:,::-1],
                                                    ),axis=1), 
                                      -random_phase[::-1,::-1],
                                    ),axis=0),
                                    ),axis=1)
    

    if mode == 'image':
        gaussian_modulus = np.abs(np.fft.fftshift(np.fft.fft2(target)))
        gaussian_field = np.fft.ifft2(np.fft.fftshift(gaussian_modulus*np.exp(1j*2*np.pi*gaussian_phase)))
    if mode == 'func':
        X = np.arange(0,M)
        Y = np.arange(0,N)
        Xgrid, Ygrid = np.meshgrid(X,Y)
        R = ((Xgrid-M/2)**2+(Ygrid-N/2)**2)**0.5
        gaussian_modulus = target(R)
        gaussian_modulus[M//2, N//2] = 0
        gaussian_field = np.fft.ifft2(np.fft.fftshift(gaussian_modulus*gaussian_phase))
        
    data = np.fft.fftshift(np.real(gaussian_field))
    return data

