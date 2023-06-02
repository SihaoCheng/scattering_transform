import numpy as np
import torch
import torch.fft
from scattering.FiltersSet import FiltersSet

class Scattering2d(object):
    def __init__(
        self, M, N, J, L=4, device='gpu', 
        wavelets='morlet', filters_set=None, weight=None, 
        precision='single', ref=None, ref_a=None, ref_b=None,
        l_oversampling=1, frequency_factor=1
    ):
        '''
        M: int (positive)
            the number of pixels along x direction
        N: int (positive)
            the number of pixels along y direction
        J: int (positive)
            the number of dyadic scales used for scattering analysis. 
            It is at most int(log2(min(M,N))) - 1.
        L: int (positive)
            the number of orientations used for oriented wavelets; 
            or the number of harmonics used for harmonic wavelets (L=1 means only monopole is used).
        device: str ('gpu' or 'cpu')
            the device to compute
        wavelets: str
            type of wavelets, can be one of the following:
            'morlet': morlet wavelet (basically off-center gaussians in Fourier space)
            'BS'    : bump-steerable wavelets (see https://arxiv.org/pdf/1810.12136.pdf)
            'gau'   : with the same angular dependence as the bump-steerable, but the radial
                    profile as radial = (2*k/k0)**2 * np.exp(-k**2/(2 * (k0/1.4)**2)). It is 
                    similar to morlet wavelet in radial profile but has a more uniform
                    orientation coverage.
            'shannon': shannon wavelets, (top-hat profiles in Fourier space)
            'gau_harmonic': its radial profile is the same as 'gau', while the orientation 
                    profile is cyclic Fourier modes (harmonics).
        filters_set : None or dict
            if None, then it is generated automatically by the parameter provided
            otherwise, it should be a dictionary with {'psi', 'phi'}, where 'psi'
                is a torch tensor with size [J, L, M, N].
        weight: numpy array or torch tensor with size [M, N]
        precision: str ('single' or 'double')
        ref: None or numpy array or torch tensor with size [N_image, M, N] 
            the reference image used to normalize the scattering covariance. 
        ref_a, ref_b: None or numpy array or torch tensor with size [N_image, M, N]
            the reference images used to normalized the 2-field scattering covariance.
        '''
        if not torch.cuda.is_available(): device='cpu'
        if filters_set is None:
            if wavelets in ['morlet', 'BS', 'gau', 'shannon']:
                filters_set = FiltersSet(M=M, N=N, J=J, L=L).generate_wavelets(
                    wavelets=wavelets, precision=precision, 
                    l_oversampling=l_oversampling, 
                    frequency_factor=frequency_factor
                )
            if wavelets=='gau_harmonic':
                filters_set = FiltersSet(M=M, N=N, J=J, L=L).generate_gau_harmonic(
                    precision=precision, frequency_factor=frequency_factor
                )
            self.M, self.N = M, N
        else: self.M, self.N = filters_set['psi'][0][0].shape
        self.J, self.L = J, L
        self.frequency_factor = frequency_factor
        self.l_oversampling = l_oversampling
        self.wavelets = wavelets
        self.precision = precision
        
        # filters set in arrays
        dtype = filters_set['psi'][0][0].dtype
        self.filters_set = torch.zeros((J,L,self.M,self.N), dtype=dtype)
        if len(filters_set['psi'][0]) == 1:
            for j in range(J):
                for l in range(L):
                    self.filters_set[j,l] = filters_set['psi'][j*L+l][0]
        else:
            self.filters_set = filters_set['psi']
        self.phi = filters_set['phi']
        
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
                    cut_high_k_off(self.weight_f, dx, dy),
                    dim=(-2,-1)
                ).real
                if device=='gpu':
                    weight_downsample = weight_downsample.cuda()
                self.weight_downsample_list.append(
                    weight_downsample / weight_downsample.mean()
                )
        self.edge_masks = get_edge_masks(M, N, J)
        
        # device
        self.device = device
        if device=='gpu':
            self.filters_set = self.filters_set.cuda()
            self.phi = self.phi.cuda()
            if weight is not None:
                self.weight = self.weight.cuda()
            self.edge_masks = self.edge_masks.cuda()
        self.edge_masks_f = torch.fft.fftn(self.edge_masks, dim=(-2,-1))
        
        # reference coefficients for cov normalization
        if ref is not None:
            self.add_ref(ref)
        if ref_b is not None:
            self.add_ref_ab(ref_a, ref_b)
    
    def add_ref(self, ref):
        self.ref_scattering_cov = self.scattering_cov(ref, if_large_batch=True)
    
    def add_ref_ab(self, ref_a, ref_b):
        self.ref_scattering_cov_2fields = self.scattering_cov_2fields(
                ref_a, ref_b, if_large_batch=True
        )
    
    def add_synthesis_P00(self, P00=None, s_cov=None, if_iso=True):
        J = self.J
        L = self.L
        self.ref_scattering_cov = {}
        if P00 is not None:
            self.ref_scattering_cov['P00'] = P00
        else:
            if if_iso:
                self.ref_scattering_cov['P00'] = torch.exp(s_cov[:,1:1+J].reshape((-1,J,1)))
            else: 
                self.ref_scattering_cov['P00'] = torch.exp(s_cov[:,1:1+J*L].reshape((-1,J,L)))
        if self.device=='gpu':
            self.ref_scattering_cov['P00'] = self.ref_scattering_cov['P00'].cuda()
            
    def add_synthesis_P11(self, s_cov, if_iso, C11_criteria='j2>=j1'):
        J = self.J
        L = self.L
        self.ref_scattering_cov = {}
        if if_iso:
            j1, j2, l2 = torch.meshgrid(torch.arange(J), torch.arange(J), torch.arange(L), indexing='ij')
            select_j12_iso = (j1 <= j2) * eval(C11_criteria)
            self.ref_scattering_cov['P11'] = torch.zeros(s_cov.shape[0], J,J,L,L)
            for i in range(select_j12_iso.sum()):
                self.ref_scattering_cov['P11'][:,j1[select_j12_iso][i],j2[select_j12_iso][i],:,l2[select_j12_iso][i]] = \
                    torch.exp(s_cov[:,1+2*J+i,None])
        else:
            j1, j2, l1, l2 = torch.meshgrid(torch.arange(J), torch.arange(J), torch.arange(L), torch.arange(L), indexing='ij')
            select_j12 = (j1 <= j2) * eval(C11_criteria)
            self.ref_scattering_cov['P11'] = torch.zeros(s_cov.shape[0], J,J,L,L)
            for i in range(select_j12.sum()):
                self.ref_scattering_cov['P11'][
                    :,j1[select_j12][i],j2[select_j12][i],l1[select_j12][i],l2[select_j12][i]
                ] = torch.exp(s_cov[:,1+2*J*L+i])
        if self.device=='gpu':
            self.ref_scattering_cov['P11'] = self.ref_scattering_cov['P11'].cuda()
        


    # ---------------------------------------------------------------------------
    #
    # scattering coefficients (mean of scattering fields) without synthesis
    #
    # ---------------------------------------------------------------------------
    def scattering_coef_simple(
        self, data, if_large_batch=False, j1j2_criteria='j2>=j1', 
        pseudo_coef=1, 
    ):
        M, N, J, L = self.M, self.N, self.J, self.L
        N_image = data.shape[0]
        filters_set = self.filters_set
        weight = self.weight

        # convert numpy array input into torch tensors
        if type(data) == np.ndarray:
            data = torch.from_numpy(data)

        # initialize tensors for scattering coefficients
        S0 = torch.zeros((N_image,1), dtype=data.dtype)
        P00= torch.zeros((N_image,J,L), dtype=data.dtype)
        S1 = torch.zeros((N_image,J,L), dtype=data.dtype)
        S2 = torch.zeros((N_image,J,J,L,L), dtype=data.dtype) + np.nan
        S2_iso = torch.zeros((N_image,J,J,L), dtype=data.dtype)
        
        # move torch tensors to gpu device, if required
        if self.device=='gpu':
            data = data.cuda()
            S0 = S0.cuda()
            P00=P00.cuda()
            S1 = S1.cuda()
            S2 = S2.cuda()
            S2_iso = S2_iso.cuda()
        
        # 0th order
        S0[:,0] = data.mean((-2,-1))
        
        # 1st and 2nd order
        data_f = torch.fft.fftn(data, dim=(-2,-1))

        if not if_large_batch:
            # only use the low-k Fourier coefs when calculating large-j scattering coefs.
            for j1 in np.arange(J):
                # 1st order: cut high k
                dx1, dy1 = self.get_dxdy(j1)
                data_f_small = cut_high_k_off(data_f, dx1, dy1)
                wavelet_f = cut_high_k_off(filters_set[j1], dx1, dy1)
                _, M1, N1 = wavelet_f.shape
                # I1(j1, l1) = | I * psi(j1, l1) |, "*" means convolution
                I1 = torch.fft.ifftn(
                    data_f_small[:,None,:,:] * wavelet_f[None,:,:,:],
                    dim=(-2,-1),
                ).abs()
                if weight is None:
                    weight_temp = 1
                else:
                    weight_temp = self.weight_downsample_list[j1][None,None,:,:]
                # S1 = I1 averaged over (x,y)
                S1 [:,j1] = (I1**pseudo_coef * weight_temp).mean((-2,-1)) * M1*N1/M/N
                P00[:,j1] = (I1**2 * weight_temp).mean((-2,-1)) * (M1*N1/M/N)**2
                # 2nd order
                I1_f = torch.fft.fftn(I1, dim=(-2,-1))
                del I1
                for j2 in np.arange(J):
                    if eval(j1j2_criteria):
                        # cut high k
                        dx2, dy2 = self.get_dxdy(j2)
                        I1_f_small = cut_high_k_off(I1_f, dx2, dy2)
                        wavelet_f2 = cut_high_k_off(filters_set[j2], dx2, dy2)
                        _, M2, N2 = wavelet_f2.shape
                        # I1(j1, l1, j2, l2) = | I1(j1, l1) * psi(j2, l2) |
                        #                    = || I * psi(j1, l1) | * psi(j2, l2)| 
                        # "*" means convolution
                        I2 = torch.fft.ifftn(
                            I1_f_small[:,:,None,:,:] * wavelet_f2[None,None,:,:,:], 
                            dim=(-2,-1),
                        ).abs()
                        if weight is None:
                            weight_temp = 1
                        else:
                            weight_temp = self.weight_downsample_list[j2][None,None,None,:,:]
                        # S2 = I2 averaged over (x,y)
                        S2[:,j1,j2,:,:] = (
                            I2**pseudo_coef * weight_temp
                        ).mean((-2,-1)) * M2*N2/M/N
        elif if_large_batch:
            # run for loop over l1 and l2, instead of calculating them all together
            # in an array. This way saves memory, but reduces the speed for small batch
            # size.
            for j1 in np.arange(J):
                # cut high k
                dx1, dy1 = self.get_dxdy(j1)
                data_f_small = cut_high_k_off(data_f, dx1, dy1)
                wavelet_f = cut_high_k_off(filters_set[j1], dx1, dy1)
                _, M1, N1 = wavelet_f.shape
                for l1 in range(L):
                    # 1st order
                    I1 = torch.fft.ifftn(
                        data_f_small * wavelet_f[None,l1], 
                        dim=(-2,-1)
                    ).abs()
                    if weight is None:
                        weight_temp = 1
                    else:
                        weight_temp = self.weight_downsample_list[j1][None,:,:]
                    S1 [:,j1,l1] = (I1**pseudo_coef * weight_temp).mean((-2,-1)) * (M1*N1/M/N)
                    P00[:,j1,l1] = (I1**2 * weight_temp).mean((-2,-1)) * (M1*N1/M/N)**2
                    # 2nd order
                    I1_f = torch.fft.fftn(I1, dim=(-2,-1))
                    del I1
                    for j2 in np.arange(J):
                        if eval(j1j2_criteria):
                            # cut high k
                            dx2, dy2 = self.get_dxdy(j2)
                            I1_f_small = cut_high_k_off(I1_f, dx2, dy2)
                            wavelet_f2 = cut_high_k_off(filters_set[j2], dx2, dy2)
                            _, M2, N2 = wavelet_f2.shape
                            for l2 in range(L):
                                I2 = torch.fft.ifftn(I1_f_small * wavelet_f2[None,l2], dim=(-2,-1)).abs()
                                if weight is None:
                                    weight_temp = 1
                                else:
                                    weight_temp = self.weight_downsample_list[j2][None,:,:]
                                S2[:,j1,j2,l1,l2] = (
                                    I2**pseudo_coef * weight_temp
                                ).mean((-2,-1)) * M2*N2/M/N

        # average over l1
        S1_iso =  S1.mean(-1)
        P00_iso= P00.mean(-1)
        for l1 in range(L):
            for l2 in range(L):
                S2_iso [:,:,:,(l2-l1)%L] += S2 [:,:,:,l1,l2]
        S2_iso  /= L
        
        # define two reduced s2 coefficients
        s21 = S2_iso.mean(-1) / S1_iso[:,:,None]
        s22 = S2_iso[:,:,:,0] / S2_iso[:,:,:,L//2]
        
        return {'S0':S0,  
                'S1_iso':  S1_iso, 
                'S2_iso':  S2_iso, 's21':s21, 's22':s22,
                'P00_iso':P00_iso,
        }
        
    
    
    
    
    # ---------------------------------------------------------------------------
    #
    # scattering coefficients (mean of scattering fields)
    #
    # ---------------------------------------------------------------------------
    def scattering_coef(
        self, data, if_large_batch=False, j1j2_criteria='j2>=j1', algorithm='fast', 
        pseudo_coef=1, remove_edge=False,
    ):
        M, N, J, L = self.M, self.N, self.J, self.L
        N_image = data.shape[0]
        filters_set = self.filters_set
        weight = self.weight

        # convert numpy array input into torch tensors
        if type(data) == np.ndarray:
            data = torch.from_numpy(data)

        # initialize tensors for scattering coefficients
        S0 = torch.zeros((N_image,1), dtype=data.dtype)
        P00= torch.zeros((N_image,J,L), dtype=data.dtype)
        S1 = torch.zeros((N_image,J,L), dtype=data.dtype)
        S2 = torch.zeros((N_image,J,J,L,L), dtype=data.dtype) + np.nan
        P11= torch.zeros((N_image,J,J,L,L), dtype=data.dtype) + np.nan
        S2_iso = torch.zeros((N_image,J,J,L), dtype=data.dtype)
        P11_iso= torch.zeros((N_image,J,J,L), dtype=data.dtype)
        E_residual = torch.zeros((N_image,J,J,L,L), dtype=data.dtype)
        
        # move torch tensors to gpu device, if required
        if self.device=='gpu':
            data = data.cuda()
            S0 = S0.cuda()
            P00=P00.cuda()
            S1 = S1.cuda()
            S2 = S2.cuda()
            P11=P11.cuda()
            S2_iso = S2_iso.cuda()
            P11_iso=P11_iso.cuda()
            E_residual = E_residual.cuda()
        
        # 0th order
        S0[:,0] = data.mean((-2,-1))
        
        # 1st and 2nd order
        data_f = torch.fft.fftn(data, dim=(-2,-1))

        if algorithm == 'fast':
            if not if_large_batch:
                # only use the low-k Fourier coefs when calculating large-j scattering coefs.
                for j1 in np.arange(J):
                    # 1st order: cut high k
                    dx1, dy1 = self.get_dxdy(j1)
                    data_f_small = cut_high_k_off(data_f, dx1, dy1)
                    wavelet_f = cut_high_k_off(filters_set[j1], dx1, dy1)
                    _, M1, N1 = wavelet_f.shape
                    # I1(j1, l1) = | I * psi(j1, l1) |, "*" means convolution
                    I1 = torch.fft.ifftn(
                        data_f_small[:,None,:,:] * wavelet_f[None,:,:,:],
                        dim=(-2,-1),
                    ).abs()
                    if weight is None:
                        weight_temp = 1
                    else:
                        weight_temp = self.weight_downsample_list[j1][None,None,:,:]
                    # S1 = I1 averaged over (x,y)
                    S1 [:,j1] = (I1**pseudo_coef * weight_temp).mean((-2,-1)) * M1*N1/M/N
                    P00[:,j1] = (I1**2 * weight_temp).mean((-2,-1)) * (M1*N1/M/N)**2
                    # 2nd order
                    I1_f = torch.fft.fftn(I1, dim=(-2,-1))
                    del I1
                    for j2 in np.arange(J):
                        if eval(j1j2_criteria):
                            # cut high k
                            dx2, dy2 = self.get_dxdy(j2)
                            I1_f_small = cut_high_k_off(I1_f, dx2, dy2)
                            wavelet_f2 = cut_high_k_off(filters_set[j2], dx2, dy2)
                            _, M2, N2 = wavelet_f2.shape
                            # I1(j1, l1, j2, l2) = | I1(j1, l1) * psi(j2, l2) |
                            #                    = || I * psi(j1, l1) | * psi(j2, l2)| 
                            # "*" means convolution
                            I2 = torch.fft.ifftn(
                                I1_f_small[:,:,None,:,:] * wavelet_f2[None,None,:,:,:], 
                                dim=(-2,-1),
                            ).abs()
                            if weight is None:
                                weight_temp = 1
                            else:
                                weight_temp = self.weight_downsample_list[j2][None,None,None,:,:]
                            # S2 = I2 averaged over (x,y)
                            S2[:,j1,j2,:,:] = (
                                I2**pseudo_coef * weight_temp
                            ).mean((-2,-1)) * M2*N2/M/N
                            P11[:,j1,j2,:,:] = (
                                I2**2 * weight_temp
                            ).mean((-2,-1)) * (M2*N2/M/N)**2
                            E_residual[:,j1,j2,:,:] = (
                                (I2 - I2.mean((-2,-1))[:,:,:,None,None])**2 * weight_temp
                            ).mean((-2,-1)) * (M2*N2/M/N)**2
            elif if_large_batch:
                # run for loop over l1 and l2, instead of calculating them all together
                # in an array. This way saves memory, but reduces the speed for small batch
                # size.
                for j1 in np.arange(J):
                    # cut high k
                    dx1, dy1 = self.get_dxdy(j1)
                    data_f_small = cut_high_k_off(data_f, dx1, dy1)
                    wavelet_f = cut_high_k_off(filters_set[j1], dx1, dy1)
                    _, M1, N1 = wavelet_f.shape
                    for l1 in range(L):
                        # 1st order
                        I1 = torch.fft.ifftn(
                            data_f_small * wavelet_f[None,l1], 
                            dim=(-2,-1)
                        ).abs()
                        if weight is None:
                            weight_temp = 1
                        else:
                            weight_temp = self.weight_downsample_list[j1][None,:,:]
                        S1 [:,j1,l1] = (I1**pseudo_coef * weight_temp).mean((-2,-1)) * (M1*N1/M/N)
                        P00[:,j1,l1] = (I1**2 * weight_temp).mean((-2,-1)) * (M1*N1/M/N)**2
                        # 2nd order
                        I1_f = torch.fft.fftn(I1, dim=(-2,-1))
                        del I1
                        for j2 in np.arange(J):
                            if eval(j1j2_criteria):
                                # cut high k
                                dx2, dy2 = self.get_dxdy(j2)
                                I1_f_small = cut_high_k_off(I1_f, dx2, dy2)
                                wavelet_f2 = cut_high_k_off(filters_set[j2], dx2, dy2)
                                _, M2, N2 = wavelet_f2.shape
                                for l2 in range(L):
                                    I2 = torch.fft.ifftn(I1_f_small * wavelet_f2[None,l2], dim=(-2,-1)).abs()
                                    if weight is None:
                                        weight_temp = 1
                                    else:
                                        weight_temp = self.weight_downsample_list[j2][None,:,:]
                                    S2[:,j1,j2,l1,l2] = (
                                        I2**pseudo_coef * weight_temp
                                    ).mean((-2,-1)) * M2*N2/M/N
                                    P11[:,j1,j2,l1,l2] = (
                                        I2**2 * weight_temp
                                    ).mean((-2,-1)) * (M2*N2/M/N)**2
                                    E_residual[:,j1,j2,l1,l2] = (
                                        (I2 - I2.mean((-2,-1))[:,None,None])**2 * weight_temp
                                    ).mean((-2,-1)) * (M2*N2/M/N)**2
        elif algorithm == 'classic':
            # I do not write the memory-friendly version here, because this "classic"
            # algorithm is just for verification purpose.
            if weight is None:
                weight_temp = 1
            else:
                weight_temp = weight[None,:,:]
                
            # 1nd order
            I1 = torch.fft.ifftn(
                data_f[:,None,None,:,:] * filters_set[None,:J,:,:,:],
                dim=(-2,-1),
            ).abs()
            S1 = (I1**pseudo_coef * weight_temp).mean((-2,-1))
            P00= (I1**2 * weight_temp).mean((-2,-1))
            I1_f = torch.fft.fftn(I1, dim=(-2,-1))

            # 2nd order
            for j1 in range(J):
                for j2 in range(J):
                    if eval(j1j2_criteria):
                        # scattering field
                        I2 = torch.fft.ifftn(
                            I1_f[:,j1,:,None,:,:] * filters_set[None,j2,None,:,:,:], 
                            dim=(-2,-1),
                        ).abs()
                        # coefficients
                        S2 [:,j1,j2,:,:] = (I2**pseudo_coef * weight_temp).mean((-2,-1))
                        P11[:,j1,j2,:,:] = (I2**2 * weight_temp).mean((-2,-1))
                        E_residual[:,j1,j2,:,:] = (
                            (I2 - I2.mean((-2,-1))[:,:,:,None,None])**2
                            * weight_temp
                        ).mean((-2,-1))

        # average over l1
        S1_iso =  S1.mean(-1)
        P00_iso= P00.mean(-1)
        for l1 in range(L):
            for l2 in range(L):
                S2_iso [:,:,:,(l2-l1)%L] += S2 [:,:,:,l1,l2]
                P11_iso[:,:,:,(l2-l1)%L] += P11[:,:,:,l1,l2]
        S2_iso  /= L
        P11_iso /= L
        
        # define two reduced s2 coefficients
        s21 = S2_iso.mean(-1) / S1_iso[:,:,None]
        s22 = S2_iso[:,:,:,0] / S2_iso[:,:,:,L//2]
        
        # get a flatten vector with size [N_image, -1] for synthesis
        for_synthesis = torch.cat((
            S1.reshape((N_image, -1)).log(), 
            S2[:,S2[0].abs()>-99].log()
        ), dim=-1)
        for_synthesis_iso = torch.cat((
            S1_iso.reshape((N_image, -1)).log(), 
            S2_iso[:,S2_iso[0].abs()>-99].log()
        ), dim=-1)
        
        return {'var': data.var((-2,-1))[:,None], 'mean': data.mean((-2,-1))[:,None],
                'S0':S0,  
                'S1':S1,  'S1_iso':  S1_iso, 
                'S2':S2,  'S2_iso':  S2_iso, 's21':s21, 's22':s22,
                'P00':P00,'P00_iso':P00_iso,
                'P11':P11,'P11_iso':P11_iso,
                'for_synthesis':for_synthesis, 'for_synthesis_iso': for_synthesis_iso,
        }
    
    # self.scattering_mean = self.scattering_coef
    
    # ---------------------------------------------------------------------------
    #
    # scattering cov
    #
    # ---------------------------------------------------------------------------
    def scattering_cov(
        self, data, if_large_batch=False, C11_criteria=None, 
        use_ref=False, normalization='P00', remove_edge=False,
        pseudo_coef=1, get_variance=False,
    ):
        '''
        Calculates the scattering correlations for a batch of images, including:
        orig. x orig.:     
                        P00 = <(I * psi)(I * psi)*> = L2(I * psi)^2
        orig. x modulus:   
                        C01 = <(I * psi2)(|I * psi1| * psi2)*> / factor
            when normalization == 'P00', factor = L2(I * psi2) * L2(I * psi1)
            when normalization == 'P11', factor = L2(I * psi2) * L2(|I * psi1| * psi2)
        modulus x modulus: 
                        C11_pre_norm = <(|I * psi1| * psi3)(|I * psi2| * psi3)>
                        C11 = C11_pre_norm / factor
            when normalization == 'P00', factor = L2(I * psi1) * L2(I * psi2)
            when normalization == 'P11', factor = L2(|I * psi1| * psi3) * L2(|I * psi2| * psi3)
        modulus x modulus (auto): 
                        P11 = <(|I * psi1| * psi2)(|I * psi1| * psi2)*>
        Parameters
        ----------
        data : numpy array or torch tensor
            image set, with size [N_image, x-sidelength, y-sidelength]
        if_large_batch : Bool (=False)
            It is recommended to use "False" unless one meets a memory issue
        C11_criteria : str or None (=None)
            Only C11 coefficients that satisfy this criteria will be computed.
            Any expressions of j1, j2, and j3 that can be evaluated as a Bool 
            is accepted.The default "None" corresponds to "j1 <= j2 <= j3".
        use_ref : Bool (=False)
            When normalizing, whether or not to use the normalization factor
            computed from a reference field. For just computing the statistics,
            the default is False. However, for synthesis, set it to "True" will
            stablize the optimization process.
        normalization : str 'P00' or 'P11' (='P00')
            Whether 'P00' or 'P11' is used as the normalization factor for C01
            and C11.
        remove_edge : Bool (=False)
            If true, the edge region with a width of rougly the size of the largest
            wavelet involved is excluded when taking the global average to obtain
            the scattering coefficients.
        
        Returns
        -------
        dict{'mean', 'var', 
            'P00', 'P00', 'S1', 'S1_iso', 'C01', 'C01_iso', 'C11', 'C11_iso', 
            'C11_pre_norm', 'C11_pre_norm_iso', 'P11', 'P11_iso',
            'for_synthesis', 'for_synthesis_iso', 
            'index_for_synthesis', 'index_for_synthesis_iso' 
        }:
        a dictionary containing different sets of scattering covariance coefficients.
        'P00'       : torch tensor with size [N_image, J, L] (# image, j1, l1)
            the power in each wavelet bands (the orig. x orig. term)
        'P00_iso'   : torch tensor with size [N_image, J] (# image, j1)
            'P00' averaged over the last dimension (l1)
        'S1'        : torch tensor with size [N_image, J, L] (# image, j1, l1)
            the 1st-order scattering coefficients, i.e., the mean of wavelet modulus fields
        'S1_iso'    : torch tensor with size [N_image, J] (# image, j1)
            'S1' averaged over the last dimension
        'C01'       : torch tensor with size [N_image, J, J, L, L] (# image, j1, j2, l1, l2)
            the orig. x modulus terms. Elements with j1 < j2 are all set to np.nan and not computed.
        'C01_iso'   : torch tensor with size [N_image, J, J, L] (# image, j1, j2, l2-l1)
            'C01' averaged over l1 while keeping l2-l1 constant.
        'C11'       : torch tensor with size [N_image, J, J, J, L, L, L] (# image, j1, j2, j3, l1, l2, l3)
            the modulus x modulus terms. Elements not satisfying j1 <= j2 <= j3 and the conditions
            defined in 'C11_criteria' are all set to np.nan and not computed.
        'C11_iso    : torch tensor with size [N_image, J, J, J, L, L] (# image, j1, j2, j3, l2-l1, l3-l1)
            'C11' averaged over l1 while keeping l2-l1 and l3-l1 constant.
        'C11_pre_norm' and 'C11_pre_norm_iso': pre-normalized modulus x modulus terms.
        'P11'       : torch tensor with size [N_image, J, J, L, L] (# image, j1, j2, l1, l2)
            the modulus x modulus terms with the two wavelets within modulus the same. Elements not following
            j1 <= j3 are set to np.nan and not computed.
        'P11_iso'   : torch tensor with size [N_image, J, J, L] (# image, j1, j2, l2-l1)
            'P11' averaged over l1 while keeping l2-l1 constant.
        'for_synthesis' : torch tensor with size [N_image, -1] (# image, index of coef.)
            flattened coefficients, containing mean/std, log(P00), log(S1), C01, and C11
        'for_synthesis_iso' : torch tensor with size [N_image, -1] (# image, index of coef.)
            flattened coefficients, containing mean/std, log(P00_iso), log(S1_iso), C01_iso, and C11_iso
        'index_for_synthesis' : torch tensor with size [7, -1] (index name, index of coef.)
            the index of the flattened tensor "for_synthesis", can be used to select subset of coef.
            the rows have the following meanings:
                index_type, j1, j2, j3, l1, l2, l3 = index_for_synthesis[:]
                where index_type is encoded by integers in the following way:
                    0: mean/std     1: log(P00)     2: log(S1)      
                    3: C01_real     4: C01_imag     5: C11_real     6: C011_imag
                    (7: P11)
                j range from 0 to J, l range from 0 to L.
        'index_for_synthesis_iso' : torch tensor with size [7, -1] (index name, index of coef.)
            same as 'index_for_synthesis_iso' except that it is for isotropic coefficients.
        '''
        if C11_criteria is None:
            C11_criteria = 'j2>=j1'
            
        M, N, J, L = self.M, self.N, self.J, self.L
        N_image = data.shape[0]
        filters_set = self.filters_set
        weight = self.weight
        if use_ref:
            if normalization=='P00': ref_P00 = self.ref_scattering_cov['P00']
            else: ref_P11 = self.ref_scattering_cov['P11']

        # convert numpy array input into torch tensors
        if type(data) == np.ndarray:
            data = torch.from_numpy(data)
            
        if self.device=='gpu':
            data = data.cuda()
        data_f = torch.fft.fftn(data, dim=(-2,-1))
        
        # initialize tensors for scattering coefficients
        P00= torch.zeros((N_image,J,L), dtype=data.dtype)
        S1 = torch.zeros((N_image,J,L), dtype=data.dtype)
        C01 = torch.zeros((N_image,J,J,L,L), dtype=data_f.dtype) + np.nan
        P11 = torch.zeros((N_image,J,J,L,L), dtype=data.dtype) + np.nan
        C11_pre_norm = torch.zeros((N_image,J,J,J,L,L,L), dtype=data_f.dtype) + np.nan
        C11 = torch.zeros((N_image,J,J,J,L,L,L), dtype=data_f.dtype) + np.nan
        
        C01_iso = torch.zeros((N_image,J,J,L), dtype=data.dtype)
        P11_iso = torch.zeros((N_image,J,J,L), dtype=data.dtype)
        C11_pre_norm_iso = torch.zeros((N_image,J,J,J,L,L), dtype=data.dtype)
        C11_iso = torch.zeros((N_image,J,J,J,L,L), dtype=data.dtype)
        
        # move torch tensors to gpu device, if required
        if self.device=='gpu':
            P00       = P00.cuda()
            S1        = S1.cuda()
            C01       = C01.cuda()
            P11       = P11.cuda()
            C11_pre_norm=C11_pre_norm.cuda()
            C11       = C11.cuda()
            C01_iso   = C01_iso.cuda()
            P11_iso   = P11_iso.cuda()
            C11_pre_norm_iso=C11_pre_norm_iso.cuda()
            C11_iso   = C11_iso.cuda()
        # variance
        if get_variance:
            P00_sigma= torch.zeros((N_image,J,L), dtype=data.dtype)
            S1_sigma = torch.zeros((N_image,J,L), dtype=data.dtype)
            C01_sigma = torch.zeros((N_image,J,J,L,L), dtype=data_f.dtype) + np.nan
            C11_sigma = torch.zeros((N_image,J,J,J,L,L,L), dtype=data_f.dtype) + np.nan
            if self.device=='gpu':
                P00       = P00.cuda()
                S1        = S1.cuda()
                C01       = C01.cuda()
                C11       = C11.cuda()
        
        # calculate scattering fields
        I1 = torch.fft.ifftn(
            data_f[:,None,None,:,:] * filters_set[None,:J,:,:,:], dim=(-2,-1)
        ).abs()
        I1_f= torch.fft.fftn(I1, dim=(-2,-1))
        
        #
        if remove_edge: 
            edge_mask = self.edge_masks[:,None,:,:]
            edge_mask = edge_mask / edge_mask.mean((-2,-1))[:,:,None,None]
        else: 
            edge_mask = 1
        P00 = (I1**2 * edge_mask).mean((-2,-1))
        S1  = (I1 * edge_mask).mean((-2,-1))
#         if get_variance:
#             P00_sigma = (I1**2 * edge_mask).std((-2,-1))
#             S1_sigma  = (I1 * edge_mask).std((-2,-1))
            
        if pseudo_coef != 1:
            I1 = I1**pseudo_coef
        
        # calculate the covariance and correlations of the scattering fields
        # only use the low-k Fourier coefs when calculating large-j scattering coefs.
        for j3 in range(0,J):
            dx3, dy3 = self.get_dxdy(j3)
            I1_f_small = cut_high_k_off(I1_f[:,:j3+1], dx3, dy3) # Nimage, J, L, x, y
            data_f_small = cut_high_k_off(data_f, dx3, dy3)
            if remove_edge:
                I1_small = torch.fft.ifftn(I1_f_small, dim=(-2,-1), norm='ortho')
                data_small = torch.fft.ifftn(data_f_small, dim=(-2,-1), norm='ortho')
            wavelet_f3 = cut_high_k_off(filters_set[j3], dx3, dy3) # L,x,y
            _, M3, N3 = wavelet_f3.shape
            wavelet_f3_squared = wavelet_f3**2
            edge_dx = min(4, int(2**j3*dx3*2/M))
            edge_dy = min(4, int(2**j3*dy3*2/N))
            # a normalization change due to the cutoff of frequency space
            fft_factor = 1 /(M3*N3) * (M3*N3/M/N)**2
            for j2 in range(0,j3+1):
                I1_f2_wf3_small = I1_f_small[:,j2].view(N_image,L,1,M3,N3) * wavelet_f3.view(1,1,L,M3,N3)
                I1_f2_wf3_2_small = I1_f_small[:,j2].view(N_image,L,1,M3,N3) * wavelet_f3_squared.view(1,1,L,M3,N3)
                if remove_edge:
                    I12_w3_small = torch.fft.ifftn(I1_f2_wf3_small, dim=(-2,-1), norm='ortho')
                    I12_w3_2_small = torch.fft.ifftn(I1_f2_wf3_2_small, dim=(-2,-1), norm='ortho')
                if use_ref:
                    if normalization=='P11':
                        norm_factor_C01 = (ref_P00[:,None,j3,:] * ref_P11[:,j2,j3,:,:]**pseudo_coef)**0.5
                    if normalization=='P00':
                        norm_factor_C01 = (ref_P00[:,None,j3,:] * ref_P00[:,j2,:,None]**pseudo_coef)**0.5
                else:
                    if normalization=='P11':
                        # [N_image,l2,l3,x,y]
                        P11_temp = (I1_f2_wf3_small.abs()**2).mean((-2,-1)) * fft_factor
                        norm_factor_C01 = (P00[:,None,j3,:] * P11_temp**pseudo_coef)**0.5
                    if normalization=='P00':
                        norm_factor_C01 = (P00[:,None,j3,:] * P00[:,j2,:,None]**pseudo_coef)**0.5

                if not remove_edge:
                    C01[:,j2,j3,:,:] = (
                        data_f_small.view(N_image,1,1,M3,N3) * torch.conj(I1_f2_wf3_small)
                    ).mean((-2,-1)) * fft_factor / norm_factor_C01
                else:
                    C01[:,j2,j3,:,:] = (
                        data_small.view(N_image,1,1,M3,N3) * torch.conj(I12_w3_small)
                    )[...,edge_dx:M3-edge_dx, edge_dy:N3-edge_dy].mean((-2,-1)) * fft_factor / norm_factor_C01
                if j2 <= j3:
                    for j1 in range(0, j2+1):
                        if eval(C11_criteria):
                            if not remove_edge:
                                if not if_large_batch:
                                    # [N_image,l1,l2,l3,x,y]
                                    C11_pre_norm[:,j1,j2,j3,:,:,:] = (
                                        I1_f_small[:,j1].view(N_image,L,1,1,M3,N3) * 
                                        torch.conj(I1_f2_wf3_2_small.view(N_image,1,L,L,M3,N3))
                                    ).mean((-2,-1)) * fft_factor
                                else:
                                    for l1 in range(L):
                                        # [N_image,l2,l3,x,y]
                                        C11_pre_norm[:,j1,j2,j3,l1,:,:] = (
                                            I1_f_small[:,j1,l1].view(N_image,1,1,M3,N3) * 
                                            torch.conj(I1_f2_wf3_2_small.view(N_image,L,L,M3,N3))
                                        ).mean((-2,-1)) * fft_factor
                            else:
                                if not if_large_batch:
                                    # [N_image,l1,l2,l3,x,y]
                                    C11_pre_norm[:,j1,j2,j3,:,:,:] = (
                                        I1_small[:,j1].view(N_image,L,1,1,M3,N3) * torch.conj(
                                            I12_w3_2_small.view(N_image,1,L,L,M3,N3)
                                        )
                                    )[...,edge_dx:-edge_dx, edge_dy:-edge_dy].mean((-2,-1)) * fft_factor
                                else:
                                    for l1 in range(L):
                                    # [N_image,l2,l3,x,y]
                                        C11_pre_norm[:,j1,j2,j3,l1,:,:] = (
                                            I1_small[:,j1].view(N_image,1,1,M3,N3) * torch.conj(
                                                I12_w3_2_small.view(N_image,L,L,M3,N3)
                                            )
                                        )[...,edge_dx:-edge_dx, edge_dy:-edge_dy].mean((-2,-1)) * fft_factor
        # define P11 from diagonals of C11
        for j1 in range(J):
            for l1 in range(L):
                P11[:,j1,:,l1,:] = C11_pre_norm[:,j1,j1,:,l1,l1,:].real
        # normalizing C11
        if normalization=='P00':
            if use_ref: P = ref_P00
            else: P = P00
            #.view(N_image,J,1,1,L,1,1) * .view(N_image,1,J,1,1,L,1)
            C11 = C11_pre_norm / (
                P[:,:,None,None,:,None,None] * P[:,None,:,None,None,:,None]
            )**(0.5*pseudo_coef)
        if normalization=='P11':
            if use_ref: P = ref_P11
            else: P = P11
            #.view(N_image,J,1,J,L,1,L) * .view(N_image,1,J,J,1,L,L)
            C11 = C11_pre_norm / (
                P[:,:,None,:,:,None,:] * P[:,None,:,:,None,:,:]
            )**(0.5*pseudo_coef)
        # average over l1 to obtain simple isotropic statistics
        P00_iso = P00.mean(-1)
        S1_iso  = S1.mean(-1)
        for l1 in range(L):
            for l2 in range(L):
                C01_iso[...,(l2-l1)%L] += C01[...,l1,l2].real
                P11_iso[...,(l2-l1)%L] += P11[...,l1,l2]
                for l3 in range(L):
                    C11_pre_norm_iso[...,(l2-l1)%L,(l3-l1)%L]+=C11_pre_norm[...,l1,l2,l3].real
                    C11_iso[...,(l2-l1)%L,(l3-l1)%L] += C11[...,l1,l2,l3].real
        C01_iso /= L; P11_iso /= L; C11_pre_norm_iso /= L; C11_iso /= L
        
        
        # get a single, flattened data vector for_synthesis
        select_and_index        = get_scattering_index(J, L, normalization, C11_criteria)
        index_for_synthesis     = select_and_index['index_for_synthesis']
        index_for_synthesis_iso = select_and_index['index_for_synthesis_iso']
        
        for_synthesis = torch.cat((
            (data.mean((-2,-1))/data.std((-2,-1)))[:,None],
            P00.reshape((N_image, -1)).log(), 
            S1.reshape((N_image, -1)).log(),
            C01[:,select_and_index['select_2']].real, 
            C01[:,select_and_index['select_2']].imag, 
            C11[:,select_and_index['select_3']].real, 
            C11[:,select_and_index['select_3']].imag
        ), dim=-1)
        for_synthesis_iso = torch.cat((
            (data.mean((-2,-1))/data.std((-2,-1)))[:,None],
            P00_iso.log(), 
            S1_iso.log(),
            C01_iso[:,select_and_index['select_2_iso']], 
#             C01_iso[:,select_and_index['select_2_iso']].imag, 
            C11_iso[:,select_and_index['select_3_iso']], 
#             C11_iso[:,select_and_index['select_3_iso']].imag
        ), dim=-1)
        if normalization=='P11':
            for_synthesis     = torch.cat(
                (for_synthesis,     P11[:,select_and_index['select_2']].log()),         
                dim=-1)
            for_synthesis_iso = torch.cat(
                (for_synthesis_iso, P11_iso[:,select_and_index['select_2_iso']].log()), 
                dim=-1)
            
        return {'var': data.var((-2,-1)), 'mean': data.mean((-2,-1)),
                'P00':P00, 'P00_iso':P00_iso,
                'S1' : S1, 'S1_iso' : S1_iso,
                'C01':C01, 'C01_iso':C01_iso,
                'C11_pre_norm':C11_pre_norm, 'C11_pre_norm_iso':C11_pre_norm_iso,
                'C11': C11,'C11_iso': C11_iso,
                'P11':P11, 'P11_iso':P11_iso,
                'for_synthesis': for_synthesis, 'for_synthesis_iso': for_synthesis_iso,
                'index_for_synthesis': index_for_synthesis,
                'index_for_synthesis_iso': index_for_synthesis_iso
        }
    
    # ---------------------------------------------------------------------------
    #
    # scattering cov for 2 fields
    #
    # ---------------------------------------------------------------------------
    def scattering_cov_2fields(
        self, data_a, data_b, if_large_batch=False, C11_criteria=None, 
        use_ref=False, normalization='P00', remove_edge=False,
    ):
        if C11_criteria is None: C11_criteria = 'j2>=j1'
            
        M, N, J, L = self.M, self.N, self.J, self.L
        N_image = data_a.shape[0]
        filters_set = self.filters_set
        weight = self.weight
        if use_ref:
            ref_P00_a = self.ref_scattering_cov_2fields['P00_a']
            ref_P00_b = self.ref_scattering_cov_2fields['P00_b']
            ref_P11_a = self.ref_scattering_cov_2fields['P11_a']
            ref_P11_b = self.ref_scattering_cov_2fields['P11_b']
        
        # convert numpy array input into torch tensors
        if type(data_a) == np.ndarray:
            data_a = torch.from_numpy(data_a)
        if type(data_b) == np.ndarray:
            data_b = torch.from_numpy(data_b)
            
        if self.device=='gpu':
            data_a = data_a.cuda()
            data_b = data_b.cuda()
        data_a_f = torch.fft.fftn(data_a, dim=(-2,-1))
        data_b_f = torch.fft.fftn(data_b, dim=(-2,-1))
        
        # initialize tensors for scattering coefficients
        P00_a = torch.zeros((N_image,J,L), dtype=data_a.dtype)
        P00_b = torch.zeros((N_image,J,L), dtype=data_a.dtype)
        C00   = torch.zeros((N_image,J,L), dtype=data_a_f.dtype)
        Corr00= torch.zeros((N_image,J,L), dtype=data_a_f.dtype)
        # S1 = torch.zeros((N_image,J,L), dtype=data.dtype)
        C01 = torch.zeros((N_image,4,J,J,L,L), dtype=data_a_f.dtype) + np.nan
        P11_a = torch.zeros((N_image,J,J,L,L), dtype=data_a.dtype) + np.nan
        P11_b = torch.zeros((N_image,J,J,L,L), dtype=data_a.dtype) + np.nan
        C11    = torch.zeros((N_image,4,J,J,J,L,L,L), dtype=data_a_f.dtype) + np.nan
        Corr11 = torch.zeros((N_image,4,J,J,J,L,L,L), dtype=data_a_f.dtype) + np.nan
        
        C01_iso = torch.zeros((N_image,4,J,J,L), dtype=data_a.dtype)
        P11_a_iso = torch.zeros((N_image,J,J,L), dtype=data_a.dtype)
        P11_b_iso = torch.zeros((N_image,J,J,L), dtype=data_a.dtype)
        C11_iso = torch.zeros((N_image,4,J,J,J,L,L), dtype=data_a.dtype)
        Corr11_iso= torch.zeros((N_image,4,J,J,J,L,L), dtype=data_a.dtype)
        
        # move torch tensors to gpu device, if required
        if self.device=='gpu':
            P00_a     = P00_a.cuda()
            P00_b     = P00_b.cuda()
            C00       = C00.cuda()
            Corr00    = Corr00.cuda()
            # S1        = S1.cuda()
            C01       = C01.cuda()
            P11_a     = P11_a.cuda()
            P11_b     = P11_b.cuda()
            C11       = C11.cuda()
            Corr11    = Corr11.cuda()
            C01_iso   = C01_iso.cuda()
            P11_a_iso = P11_a_iso.cuda()
            P11_b_iso = P11_b_iso.cuda()
            C11_iso   = C11_iso.cuda()
            Corr11_iso= Corr11_iso.cuda()
        
        # calculate scattering fields
        I1_a = torch.fft.ifftn(
            data_a_f[:,None,None,:,:] * filters_set[None,:J,:,:,:], dim=(-2,-1)
        ).abs()
        I1_b = torch.fft.ifftn(
            data_b_f[:,None,None,:,:] * filters_set[None,:J,:,:,:], dim=(-2,-1)
        ).abs()
        I1_a_f = torch.fft.fftn(I1_a, dim=(-2,-1))
        I1_b_f = torch.fft.fftn(I1_b, dim=(-2,-1))
        
        P00_a = (I1_a**2).mean((-2,-1))
        P00_b = (I1_b**2).mean((-2,-1))
        
        C00 = (
            (data_a_f * torch.conj(data_b_f))[:,None,None,:,:] * filters_set[None,:J,:,:,:]**2
        ).mean((-2,-1)) /M/N
        Corr00 = C00 / (P00_a * P00_b)**0.5
        # S1  = I1.mean((-2,-1))
        
        # calculate the covariance and correlations of the scattering fields
        # only use the low-k Fourier coefs when calculating large-j scattering coefs.
        for j3 in range(0,J):
            dx3, dy3 = self.get_dxdy(j3)
            I1_a_f_small = cut_high_k_off(I1_a_f, dx3, dy3)
            I1_b_f_small = cut_high_k_off(I1_b_f, dx3, dy3)
            data_a_f_small = cut_high_k_off(data_a_f, dx3, dy3)
            data_b_f_small = cut_high_k_off(data_b_f, dx3, dy3)
            wavelet_f3 = cut_high_k_off(filters_set[j3], dx3, dy3)
            _, M3, N3 = wavelet_f3.shape
            wavelet_f3_squared = wavelet_f3**2
            # a normalization change due to the cutoff of frequency space
            fft_factor = 1 /(M3*N3) * (M3*N3/M/N)**2
            for j2 in range(0,j3+1):
                # [N_image,l2,l3,x,y]
                P11_a_temp = (
                    I1_a_f_small[:,j2].view(N_image,L,1,M3,N3).abs()**2 * 
                    wavelet_f3_squared.view(1,1,L,M3,N3)
                ).mean((-2,-1)) * fft_factor
                P11_b_temp = (
                    I1_b_f_small[:,j2].view(N_image,L,1,M3,N3).abs()**2 * 
                    wavelet_f3_squared.view(1,1,L,M3,N3)
                ).mean((-2,-1)) * fft_factor
                if use_ref:
                    if normalization=='P00':
                        norm_factor_C01_a  = (ref_P00_a[:,None,j3,:] * ref_P00_a[:,j2,:,None])**0.5
                        norm_factor_C01_b  = (ref_P00_b[:,None,j3,:] * ref_P00_b[:,j2,:,None])**0.5
                        norm_factor_C01_ab = (ref_P00_a[:,None,j3,:] * ref_P00_b[:,j2,:,None])**0.5
                        norm_factor_C01_ba = (ref_P00_b[:,None,j3,:] * ref_P00_a[:,j2,:,None])**0.5
                    if normalization=='P11':
                        norm_factor_C01_a  = (ref_P00_a[:,None,j3,:] * ref_P11_a[:,j2,j3,:,:])**0.5
                        norm_factor_C01_b  = (ref_P00_b[:,None,j3,:] * ref_P11_b[:,j2,j3,:,:])**0.5
                        norm_factor_C01_ab = (ref_P00_a[:,None,j3,:] * ref_P11_b[:,j2,j3,:,:])**0.5
                        norm_factor_C01_ba = (ref_P00_b[:,None,j3,:] * ref_P11_a[:,j2,j3,:,:])**0.5
                else:
                    if normalization=='P00':
                        norm_factor_C01_a =  (P00_a[:,None,j3,:] * P00_a[:,j2,:,None])**0.5
                        norm_factor_C01_b =  (P00_b[:,None,j3,:] * P00_b[:,j2,:,None])**0.5
                        norm_factor_C01_ab = (P00_a[:,None,j3,:] * P00_b[:,j2,:,None])**0.5
                        norm_factor_C01_ba = (P00_b[:,None,j3,:] * P00_a[:,j2,:,None])**0.5
                    if normalization=='P11':
                        norm_factor_C01_a =  (P00_a[:,None,j3,:] * P11_a_temp)**0.5
                        norm_factor_C01_b =  (P00_b[:,None,j3,:] * P11_b_temp)**0.5
                        norm_factor_C01_ab = (P00_a[:,None,j3,:] * P11_b_temp)**0.5
                        norm_factor_C01_ba = (P00_b[:,None,j3,:] * P11_a_temp)**0.5
                C01[:,0,j2,j3,:,:] = (
                    data_a_f_small.view(N_image,1,1,M3,N3) * 
                    torch.conj(I1_a_f_small[:,j2].view(N_image,L,1,M3,N3)) *
                    wavelet_f3.view(1,1,L,M3,N3)
                ).mean((-2,-1)) * fft_factor / norm_factor_C01_a
                C01[:,1,j2,j3,:,:] = (
                    data_b_f_small.view(N_image,1,1,M3,N3) * 
                    torch.conj(I1_b_f_small[:,j2].view(N_image,L,1,M3,N3)) *
                    wavelet_f3.view(1,1,L,M3,N3)
                ).mean((-2,-1)) * fft_factor / norm_factor_C01_b
                C01[:,2,j2,j3,:,:] = (
                    data_a_f_small.view(N_image,1,1,M3,N3) * 
                    torch.conj(I1_b_f_small[:,j2].view(N_image,L,1,M3,N3)) *
                    wavelet_f3.view(1,1,L,M3,N3)
                ).mean((-2,-1)) * fft_factor / norm_factor_C01_ab
                C01[:,3,j2,j3,:,:] = (
                    data_b_f_small.view(N_image,1,1,M3,N3) * 
                    torch.conj(I1_a_f_small[:,j2].view(N_image,L,1,M3,N3)) *
                    wavelet_f3.view(1,1,L,M3,N3)
                ).mean((-2,-1)) * fft_factor / norm_factor_C01_ba
                for j1 in range(0, j2+1):
                    if eval(C11_criteria):
                        if not if_large_batch:
                            # [N_image,l1,l2,l3,x,y]
                            C11[:,0,j1,j2,j3,:,:,:] = (
                                I1_a_f_small[:,j1].view(N_image,L,1,1,M3,N3) * 
                                torch.conj(I1_a_f_small[:,j2].view(N_image,1,L,1,M3,N3)) *
                                wavelet_f3_squared.view(1,1,1,L,M3,N3)
                            ).mean((-2,-1)) * fft_factor
                            C11[:,1,j1,j2,j3,:,:,:] = (
                                I1_b_f_small[:,j1].view(N_image,L,1,1,M3,N3) * 
                                torch.conj(I1_b_f_small[:,j2].view(N_image,1,L,1,M3,N3)) *
                                wavelet_f3_squared.view(1,1,1,L,M3,N3)
                            ).mean((-2,-1)) * fft_factor
                            C11[:,2,j1,j2,j3,:,:,:] = (
                                I1_a_f_small[:,j1].view(N_image,L,1,1,M3,N3) * 
                                torch.conj(I1_b_f_small[:,j2].view(N_image,1,L,1,M3,N3)) *
                                wavelet_f3_squared.view(1,1,1,L,M3,N3)
                            ).mean((-2,-1)) * fft_factor
                            C11[:,3,j1,j2,j3,:,:,:] = (
                                I1_b_f_small[:,j1].view(N_image,L,1,1,M3,N3) * 
                                torch.conj(I1_a_f_small[:,j2].view(N_image,1,L,1,M3,N3)) *
                                wavelet_f3_squared.view(1,1,1,L,M3,N3)
                            ).mean((-2,-1)) * fft_factor
                        else:
                            for l1 in range(L):
                            # [N_image,l2,l3,x,y]
                                C11[:,0,j1,j2,j3,l1,:,:] = (
                                    I1_a_f_small[:,j1,l1].view(N_image,1,1,M3,N3) * 
                                    torch.conj(I1_a_f_small[:,j2].view(N_image,L,1,M3,N3)) *
                                    wavelet_f3_squared.view(1,1,L,M3,N3)
                                ).mean((-2,-1)) * fft_factor
                                C11[:,1,j1,j2,j3,l1,:,:] = (
                                    I1_b_f_small[:,j1,l1].view(N_image,1,1,M3,N3) * 
                                    torch.conj(I1_b_f_small[:,j2].view(N_image,L,1,M3,N3)) *
                                    wavelet_f3_squared.view(1,1,L,M3,N3)
                                ).mean((-2,-1)) * fft_factor
                                C11[:,2,j1,j2,j3,l1,:,:] = (
                                    I1_a_f_small[:,j1,l1].view(N_image,1,1,M3,N3) * 
                                    torch.conj(I1_b_f_small[:,j2].view(N_image,L,1,M3,N3)) *
                                    wavelet_f3_squared.view(1,1,L,M3,N3)
                                ).mean((-2,-1)) * fft_factor
                                C11[:,3,j1,j2,j3,l1,:,:] = (
                                    I1_a_f_small[:,j1,l1].view(N_image,1,1,M3,N3) * 
                                    torch.conj(I1_b_f_small[:,j2].view(N_image,L,1,M3,N3)) *
                                    wavelet_f3_squared.view(1,1,L,M3,N3)
                                ).mean((-2,-1)) * fft_factor
        # define P11 from C11
        for j1 in range(J):
            for l1 in range(L):
                for j3 in range(j1, J):
                    P11_a[:,j1,j3,l1,:] = C11[:,0,j1,j1,j3,l1,l1,:].real
                    P11_b[:,j1,j3,l1,:] = C11[:,1,j1,j1,j3,l1,l1,:].real
        # normalizing C11
        if normalization=='P00':
            if use_ref: 
                Pa = ref_P00_a; Pb = ref_P00_b
            else:
                Pa = P00_a; Pb = P00_b
            #.view(N_image,J,1,1,L,1,1) *.view(N_image,1,J,1,1,L,1)
            Corr11[:,0] = C11[:,0] / (Pa[:,:,None,None,:,None,None] * Pa[:,None,:,None,None,:,None])**0.5
            Corr11[:,1] = C11[:,1] / (Pb[:,:,None,None,:,None,None] * Pb[:,None,:,None,None,:,None])**0.5
            Corr11[:,2] = C11[:,2] / (Pa[:,:,None,None,:,None,None] * Pb[:,None,:,None,None,:,None])**0.5
            Corr11[:,3] = C11[:,3] / (Pb[:,:,None,None,:,None,None] * Pa[:,None,:,None,None,:,None])**0.5
        if normalization=='P11':
            if use_ref: 
                Pa = ref_P11_a; Pb = ref_P11_b
            else:
                Pa = P11_a; Pb = P11_b
            #.view(N_image,J,1,J,L,1,L) * .view(N_image,1,1,J,L,J,L)
            Corr11[:,0] = C11[:,0] / (Pa[:,:,None,:,:,None,:] * Pa[:,None,:,:,None,:,:])**0.5
            Corr11[:,1] = C11[:,1] / (Pb[:,:,None,:,:,None,:] * Pb[:,None,:,:,None,:,:])**0.5
            Corr11[:,2] = C11[:,2] / (Pa[:,:,None,:,:,None,:] * Pb[:,None,:,:,None,:,:])**0.5
            Corr11[:,3] = C11[:,3] / (Pb[:,:,None,:,:,None,:] * Pa[:,None,:,:,None,:,:])**0.5
        
        # average over l1 to obtain simple isotropic statistics
        P00_a_iso = P00_a.mean(-1)
        P00_b_iso = P00_b.mean(-1)
        Corr00_iso= Corr00.mean(-1)
        # S1_iso  = S1.mean(-1)
        for l1 in range(L):
            for l2 in range(L):
                C01_iso[...,(l2-l1)%L] += C01[...,l1,l2].real
                P11_a_iso[...,(l2-l1)%L] += P11_a[...,l1,l2]
                P11_b_iso[...,(l2-l1)%L] += P11_b[...,l1,l2]
                for l3 in range(L):
                    C11_iso   [...,(l2-l1)%L,(l3-l1)%L] +=    C11[...,l1,l2,l3].real
                    Corr11_iso[...,(l2-l1)%L,(l3-l1)%L] += Corr11[...,l1,l2,l3].real
        C01_iso /= L; P11_a_iso /= L; P11_b_iso /= L; C11_iso /= L; Corr11_iso /= L
        
        # generate single, flattened data vector for_synthesis
        select_and_index = get_scattering_index(J, L, normalization, C11_criteria, 2)
        index_for_synthesis     = select_and_index['index_for_synthesis']
        index_for_synthesis_iso = select_and_index['index_for_synthesis_iso']
        
        for_synthesis = torch.cat((
            (data_a.mean((-2,-1))/data_a.std((-2,-1)))[:,None],
            (data_b.mean((-2,-1))/data_b.std((-2,-1)))[:,None],
            P00_a.reshape((N_image, -1)).log(), 
            P00_b.reshape((N_image, -1)).log(), 
            Corr00.reshape((N_image, -1)).real, 
            Corr00.reshape((N_image, -1)).imag, 
            C01[:,:,select_and_index['select_2']].reshape((N_image, -1)).real, 
            C01[:,:,select_and_index['select_2']].reshape((N_image, -1)).imag, 
            Corr11[:,:,select_and_index['select_3']].reshape((N_image, -1)).real, 
            Corr11[:,:,select_and_index['select_3']].reshape((N_image, -1)).imag
        ), dim=-1)
        for_synthesis_iso = torch.cat((
            (data_a.mean((-2,-1))/data_a.std((-2,-1)))[:,None],
            (data_b.mean((-2,-1))/data_b.std((-2,-1)))[:,None],
            P00_a_iso.log(), 
            P00_b_iso.log(), 
            Corr00_iso.real, 
            Corr00_iso.imag,
            C01_iso[:,:,select_and_index['select_2_iso']].reshape((N_image, -1)), 
#             C01_iso[:,:,select_and_index['select_2_iso']].reshape((N_image, -1)).imag, 
            Corr11_iso[:,:,select_and_index['select_3_iso']].reshape((N_image, -1)), 
#             Corr11_iso[:,:,select_and_index['select_3_iso']].reshape((N_image, -1)).imag
        ), dim=-1)
            
        return {'var_a': data_a.var((-2,-1)), 'mean_a': data_a.mean((-2,-1)),
                'var_b': data_b.var((-2,-1)), 'mean_b': data_b.mean((-2,-1)),
                'P00_a':P00_a, 'P00_a_iso':P00_a_iso, 'P00_b':P00_b, 'P00_b_iso':P00_b_iso,
                'Corr00': Corr00, 'Corr00_iso': Corr00_iso,
                'C01':C01, 'C01_iso':C01_iso,
                'P11_a':P11_a, 'P11_a_iso':P11_a_iso, 'P11_b':P11_b, 'P11_b_iso':P11_b_iso,
                'C11':C11, 'C11_iso':C11_iso,
                'Corr11': Corr11,'Corr11_iso': Corr11_iso,
                'for_synthesis': for_synthesis, 'for_synthesis_iso': for_synthesis_iso,
                'index_for_synthesis': index_for_synthesis,
                'index_for_synthesis_iso': index_for_synthesis_iso,
        }
    
    # ---------------------------------------------------------------------------
    #
    # utility functions for computing scattering coef and covariance
    #
    # ---------------------------------------------------------------------------
     
    def get_dxdy(self, j):
        dx = int(max( 8, min( np.ceil(self.M/2**j*self.frequency_factor), self.M//2 ) ))
        dy = int(max( 8, min( np.ceil(self.N/2**j*self.frequency_factor), self.N//2 ) ))
        return dx, dy
    
    # ---------------------------------------------------------------------------
    #
    # get I1
    #
    # ---------------------------------------------------------------------------
    def get_I1(self, data, J, L, pseudo_coef=1):
        '''
            Calculates the scattering fields (activations) I1 = |I0 \star \psi(j,l)|
            Parameters
            ----------
            data : numpy array or torch tensor
                image set, with size [N_image, x-sidelength, y-sidelength]
            J, L : int
                the number of scales and angles for calculation
            pseudo_coef : float/int, optional
                the power of modulus. This allows for a modification of 
                the scattering transform. For standard ST, it should be 
                one, which is also the default. When it is 2, the n-th
                order ST will become some 2^n point functions.
            Returns
            -------
            I1 : torch tensor
                ST field I1, with size [N_image, J, L]
        '''
        M, N = self.M, self.N
        N_image = data.shape[0]
        filters_set = self.filters_set
        weight = self.weight

        # convert numpy array input into torch tensors
        if type(data) == np.ndarray:
            data = torch.from_numpy(data)

        # move torch tensors to gpu device, if required
        if self.device=='gpu':
            data = data.cuda()
        
        data_f = torch.fft.fftn(data, dim=(-2,-1))
        
        # calculating scattering coefficients, with two Fourier transforms
        if weight is None:
            weight_temp = 1
        else:
            weight_temp = weight[None,None,None,:,:]
        # 1st-order scattering field
        I1 = torch.fft.ifftn(
            data_f[:,None,None,:,:] * filters_set[None,:J,:,:,:],
            dim=(-2,-1),
        )#.abs()**pseudo_coef

        return I1

    
# ------------------------------------------------------------------------------------------
#
# end of scattering calculator
#
# ------------------------------------------------------------------------------------------




# ------------------------------------------------------------------------------------------
#
# utility functions 
#
# ------------------------------------------------------------------------------------------
def cut_high_k_off(data_f, dx, dy):
    if_xodd = (data_f.shape[-2]%2==1)
    if_yodd = (data_f.shape[-1]%2==1)
    result = torch.cat(
        (torch.cat(
            ( data_f[...,:dx+if_xodd, :dy+if_yodd] , data_f[...,-dx:, :dy+if_yodd]
            ), -2),
          torch.cat(
            ( data_f[...,:dx+if_xodd, -dy:] , data_f[...,-dx:, -dy:]
            ), -2)
        ),-1)
    return result


def get_edge_masks(M, N, J, d0=1):
    edge_masks = torch.empty((J, M, N))
    X, Y = torch.meshgrid(torch.arange(M), torch.arange(N), indexing='ij')
    for j in range(J):
        edge_dx = min(M//4, 2**j*d0)
        edge_dy = min(N//4, 2**j*d0)
        edge_masks[j] = (X>=edge_dx) * (X<=M-edge_dx) * (Y>=edge_dy) * (Y<=N-edge_dy)
    return edge_masks


def get_scattering_index(J, L, normalization='P00', C11_criteria='j1>-1', num_field=1):
    '''
    the labels of different types of coefficients are as follows:
    0: mean     1: P00      2:S1
    3:C01re     4: C01im    5: C11re    6: C11im
    (7: P11)
    '''
    # select elements
    # one-scale coef
    j1, l1 = torch.meshgrid(
        torch.arange(J), torch.arange(L), indexing='ij'
    )
    select_1 = j1 > -1
    invalid = j1[None,select_1]*0-1
    index_1 = torch.cat(
        (j1[None,select_1], invalid, invalid, l1[None,select_1], invalid, invalid),
        dim=0)
    # one-scale isotropic coef
    j1, = torch.meshgrid(torch.arange(J), indexing='ij')
    select_1_iso = j1 > -1
    invalid = j1[None,select_1_iso]*0-1
    index_1_iso = torch.cat(
        (j1[None,select_1_iso], invalid, invalid, invalid, invalid, invalid),
        dim=0)
    # two-scale coef
    j1, j2, l1, l2 = torch.meshgrid(
        torch.arange(J), torch.arange(J), 
        torch.arange(L), torch.arange(L), indexing='ij'
    )
    select_2 = j1 <= j2
    invalid = j1[None,select_2]*0-1
    index_2 = torch.cat(
        (j1[None,select_2], j2[None,select_2], invalid, 
         l1[None,select_2], l2[None,select_2], invalid),
        dim=0)
    # two-scale isotropic coef
    j1, j2, l2 = torch.meshgrid(torch.arange(J), torch.arange(J), torch.arange(L), indexing='ij')
    select_2_iso = j1 <= j2
    invalid = j1[None,select_2_iso]*0-1
    index_2_iso = torch.cat(
        (j1[None,select_2_iso], j2[None,select_2_iso], invalid, 
         invalid,               l2[None,select_2_iso], invalid),
        dim=0)
    # three-scale coef
    j1, j2, j3, l1, l2, l3 = torch.meshgrid(
        torch.arange(J), torch.arange(J), torch.arange(J), 
        torch.arange(L), torch.arange(L), torch.arange(L), indexing='ij'
    )
    if normalization=='P00' and num_field==1:
        select_3 = (j1 <= j2) * (j2 <= j3) * eval(C11_criteria)
    else:
        select_3 = (j1 <= j2) * (j2 <= j3) * eval(C11_criteria) * ~((l1==l2)*(j1==j2))
    index_3 = torch.cat(
        (j1[None,select_3], j2[None,select_3], j3[None,select_3],
         l1[None,select_3], l2[None,select_3], l3[None,select_3]),
        dim=0)
    # three-scale isotropic coef
    j1, j2, j3, l2, l3 = torch.meshgrid(
        torch.arange(J), torch.arange(J), torch.arange(J), 
        torch.arange(L), torch.arange(L), indexing='ij'
    )
    if normalization=='P00' and num_field==1:
        select_3_iso = (j1 <= j2) * (j2 <= j3) * eval(C11_criteria)
    else:
        select_3_iso = (j1 <= j2) * (j2 <= j3) * eval(C11_criteria) * ~((l2==0)*(j1==j2))
    invalid = j1[None,select_3_iso]*0-1
    index_3_iso = torch.cat(
        (j1[None,select_3_iso], j2[None,select_3_iso], j3[None,select_3_iso],
         invalid,               l2[None,select_3_iso], l3[None,select_3_iso]),
        dim=0)
    # concatenate index of all coef
    index_mean = index_1[:,0:1]*0-1
    if num_field==1:
        index_for_synthesis = torch.cat((
            index_mean, index_1, index_1, # mean, P00, S1
            index_2,    index_2, index_3, index_3, # 
        ), dim=-1)
        index_for_synthesis_iso = torch.cat((
            index_mean,  index_1_iso, index_1_iso, # mean, P00, S1
            index_2_iso,         index_3_iso, # C01re, C11re,
        ), dim=-1)
        # label different coef
        index_type = torch.cat((
            index_mean[:1,:]*0, index_1[:1,:]*0+1, index_1[:1,:]*0+2,
            index_2[:1,:]*0+3,  index_2[:1,:]*0+4, index_3[:1,:]*0+5, index_3[:1,:]*0+6
        ), dim=-1)
        index_type_iso = torch.cat((
            index_mean[:1,:]*0,    index_1_iso[:1,:]*0+1, index_1_iso[:1,:]*0+2,
            index_2_iso[:1,:]*0+3,                 index_3_iso[:1,:]*0+5,
        ), dim=-1)
        # add P11 to the coef set
        if normalization=='P11':
            index_for_synthesis     = torch.cat((index_for_synthesis    , index_2    ), dim=-1) # P11
            index_for_synthesis_iso = torch.cat((index_for_synthesis_iso, index_2_iso), dim=-1) # P11
            index_type              = torch.cat((index_type             , index_2[:1,:]*0+7), dim=-1) # P11
            index_type_iso          = torch.cat((index_type_iso         , index_2_iso[:1,:]*0+7), dim=-1) # P11
    if num_field==2:
        index_for_synthesis = torch.cat((
            index_mean, index_mean, # mean_a, mean_b,
            index_1, index_1, index_1, index_1, #P00_a, P00_b, C00re, C00im
            index_2, index_2, index_2, index_2, index_2, index_2, index_2, index_2, # C01re, im
            index_3, index_3, index_3, index_3, index_3, index_3, index_3, index_3, # Corr11 (re, im)
        ), dim=-1)
        index_for_synthesis_iso = torch.cat((
            index_mean,  index_mean, # mean_a, mean_b,
            index_1_iso, index_1_iso, index_1_iso, index_1_iso, # P00_a, P00_b, C00re, C00im
            index_2_iso, index_2_iso, index_2_iso, index_2_iso, # C01re
#                 index_2_iso, index_2_iso, index_2_iso, index_2_iso, # C01im
            index_3_iso, index_3_iso, index_3_iso, index_3_iso, # C11re
#                 index_3_iso, index_3_iso, index_3_iso, index_3_iso, # C11im
        ), dim=-1)
        # label different coef
        index_type = torch.cat((
            index_mean[:1,:]*0, index_mean[:1,:]*0+1,
            index_1[:1,:]*0+2,  index_1[:1,:]*0+3, index_1[:1,:]*0+4, index_1[:1,:]*0+5, 
            index_2[:1,:]*0+6,  index_2[:1,:]*0+7, index_2[:1,:]*0+8, index_2[:1,:]*0+9,
            index_2[:1,:]*0+10, index_2[:1,:]*0+11,index_2[:1,:]*0+12,index_2[:1,:]*0+13,
            index_3[:1,:]*0+14, index_3[:1,:]*0+15,index_3[:1,:]*0+16,index_3[:1,:]*0+17,
            index_3[:1,:]*0+18, index_3[:1,:]*0+19,index_3[:1,:]*0+20,index_3[:1,:]*0+21,

        ), dim=-1)
        index_type_iso = torch.cat((
            index_mean[:1,:]*0, index_mean[:1,:]*0+1, 
            index_1_iso[:1,:]*0+2,  index_1_iso[:1,:]*0+3, index_1_iso[:1,:]*0+4, index_1_iso[:1,:]*0+5,
            index_2_iso[:1,:]*0+6,  index_2_iso[:1,:]*0+7, index_2_iso[:1,:]*0+8, index_2_iso[:1,:]*0+9,
#                 index_2_iso[:1,:]*0+10, index_2_iso[:1,:]*0+11,index_2_iso[:1,:]*0+12,index_2_iso[:1,:]*0+13,
            index_3_iso[:1,:]*0+14, index_3_iso[:1,:]*0+15,index_3_iso[:1,:]*0+16,index_3_iso[:1,:]*0+17,
#                 index_3_iso[:1,:]*0+18, index_3_iso[:1,:]*0+19,index_3_iso[:1,:]*0+20,index_3_iso[:1,:]*0+21,
        ), dim=-1)
    # combine index and the label of coef type 
    index_for_synthesis     = torch.cat((index_type,     index_for_synthesis),     dim=0)
    index_for_synthesis_iso = torch.cat((index_type_iso, index_for_synthesis_iso), dim=0)

    return {'select_2': select_2, 'select_2_iso': select_2_iso,
            'select_3': select_3, 'select_3_iso': select_3_iso,
            'index_1': index_1, 'index_1_iso': index_1_iso, 
            'index_2': index_2, 'index_2_iso': index_2_iso, 
            'index_3': index_3, 'index_3_iso': index_3_iso,
            'index_for_synthesis': index_for_synthesis, 
            'index_for_synthesis_iso': index_for_synthesis_iso,
            }
