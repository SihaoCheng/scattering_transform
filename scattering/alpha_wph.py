
# __all__ = ['Scattering']

import warnings
import torch
import torch.fft as fft
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import math
import torch.nn.functional as F
from .backend import cdgmm, Modulus, SubsampleFourier, Pad, mulcu, \
    SubInitSpatialMeanR, SubInitSpatialMeanC, DivInitStdR, DivInitStd, \
    padc, conjugate, maskns, masks_subsample_shift3, \
    extract_shift3
from .ST import FiltersSet
#from .filter_bank import filter_bank
#from .utils import compute_padding, fft2_c2c, ifft2_c2r, ifft2_c2c, periodic_dis, \
#    periodic_signed_dis

class PhaseHarmonics2d(object):

    def __init__(
        self, M, N, J, L=4, A=4, A_prime=1, delta_j=1, delta_l=4,
        nb_chunks=1, chunk_id=0, shift='all', wavelets='morlet',
        filter_path=None,#'./filters/'
    ):
        self.M, self.N, self.J, self.L = M, N, J, L
        self.nb_chunks = nb_chunks  # number of chunks to cut whp cov
        self.chunk_id = chunk_id
        self.pre_pad = False  # no padding
        self.cache = False
        self.A = A # number of phase shifts one one side of the correlation
        self.A_prime = A_prime # number of phase shifts on the other side
        self.delta_j = delta_j # number of scales interactions
        self.delta_l = delta_l  # number of angles interactions
        self.wavelets = wavelets
        self.shift = shift
        self.path = filter_path
        print(self.wavelets)
        assert(self.chunk_id <= self.nb_chunks)
        if self.delta_l > self.L:
            raise (
                ValueError('delta_l must be <= L'))
        self.build()

    def build(self):
        self.modulus = Modulus()
        # self.pad = Pad(2**self.J, pre_pad = self.pre_pad)
        self.pad = Pad(0, pre_pad=self.pre_pad)
        self.subsample_fourier = SubsampleFourier()
        self.M_padded, self.N_padded = self.M, self.N
        lim_shift = int(math.log2(self.M)) - 3
        self.masks_shift = masks_subsample_shift3(self.J,self.M,self.N)
        self.masks_shift = torch.cat((torch.zeros(1,self.M, self.N),
                                      self.masks_shift), dim=0)
        self.masks_shift[0,0,0] = 1.
        self.factr_shift = self.masks_shift.sum(dim=(-2,-1))
        self.indices = extract_shift3(self.masks_shift[2,:,:])

        if self.wavelets == 'bump':
            self.filters_tensor()
        elif self.wavelets == 'morlet':
            self.filters_tensor_morlet()

        if self.chunk_id < self.nb_chunks:
            self.idx_wph = self.compute_idx()
            self.this_wph = self.get_this_chunk(self.nb_chunks, self.chunk_id)
            self.subinitmean1 = SubInitSpatialMeanR()
            self.subinitmean2 = SubInitSpatialMeanR()
            self.divinitstd1 = DivInitStdR()
            self.divinitstd2 = DivInitStdR()
            self.divinitstdJ = DivInitStdR()
            self.subinitmeanJ = SubInitSpatialMeanR()
            self.subinitmeanin = SubInitSpatialMeanR()
            self.divinitstdin = DivInitStdR()
            self.divinitstdH = [None,None,None]
            for hid in range(3):
                self.divinitstdH[hid] = DivInitStdR()

    def filters_tensor_morlet(self):

        J = self.J
        M = self.M; N = self.N; L = self.L
        
        if self.path is not None:
            hatpsi_ = torch.load(self.path+'morlet_N'+str(N)+'_J'+str(J)+'_L'+str(L)+'.pt') # (J,L,M,N,2)
            hatpsi = torch.cat((hatpsi_, torch.flip(hatpsi_, (2,3))), dim=1).numpy() # (J,L2,M,N,2)
            fftpsi = hatpsi[...,0] + hatpsi[...,1]* 1.0j
            hatphi = torch.load(self.path+'/morlet_lp_N'+str(N)+'_J'+str(J)+'_L'+str(L)+'.pt').numpy()  # (M,N,2)
        else:
            Sihao_filters = FiltersSet(M=N, N=N, J=J, L=L).generate_morlet()
            hatpsi_ = Sihao_filters['psi'] # (J,L,M,N)
            hatpsi_ = torch.cat((hatpsi_[...,None], hatpsi_[...,None]*0), dim=-1) # (J,L,M,N,2)
            hatpsi = torch.cat((hatpsi_, torch.flip(hatpsi_, (2,3))), dim=1).numpy() # (J,L2,M,N,2)
            fftpsi = hatpsi[...,0] + hatpsi[...,1]* 1.0j
            hatphi = Sihao_filters['phi']  # (M,N)
            hatphi = torch.cat((hatphi[...,None], hatphi[...,None]*0), dim=-1).numpy() # (M,N,2)

        
        A = self.A
        A_prime = self.A_prime

        alphas = np.arange(A, dtype=np.float32)/(max(A,1))*np.pi*2
        alphas = np.exp(1j * alphas)

        alphas_prime = np.arange(A_prime, dtype=np.float32)/(max(A_prime,1))*np.pi*2
        alphas_prime = np.exp(1j * alphas_prime)

        filt = np.zeros((J, 2*L, A, self.M, self.N), dtype=np.complex_)
        filt_prime = np.zeros((J, 2*L, A_prime, self.M, self.N),
                              dtype=np.complex_)

        for alpha in range(A):
            for j in range(J):
                for theta in range(L):
                    psi_signal = fftpsi[j, theta, ...]
                    filt[j, theta, alpha, :, :] = alphas[alpha]*psi_signal
                    filt[j, L+theta, alpha, :, :] = np.conj(alphas[alpha]) \
                        * psi_signal

        for alpha in range(A_prime):
            for j in range(J):
                for theta in range(L):
                    psi_signal = fftpsi[j, theta, ...]
                    filt_prime[j, theta, alpha, :, :] = alphas_prime[alpha]*psi_signal
                    filt_prime[j, L+theta, alpha, :, :] = np.conj(alphas_prime[alpha])*psi_signal

        filters = np.stack((np.real(filt), np.imag(filt)), axis=-1)
        filters_prime = np.stack((np.real(filt_prime), np.imag(filt_prime)),
                                 axis=-1)

        self.hatphi = torch.view_as_complex(torch.FloatTensor(hatphi)).type(torch.cfloat)  # (M,N,2)
        self.hatpsi = torch.view_as_complex(torch.FloatTensor(filters)).type(torch.cfloat)
        self.hatpsi_prime = torch.view_as_complex(torch.FloatTensor(filters_prime)).type(torch.cfloat)

        # add haar
        self.hathaar2d = torch.view_as_complex(torch.zeros(3,M,N,2))
        psi = torch.zeros(M,N,2)
        psi[1,1,1] = 1/4
        psi[1,2,1] = -1/4
        psi[2,1,1] = 1/4
        psi[2,2,1] = -1/4
        self.hathaar2d[0,:,:] = fft.fft2(torch.view_as_complex(psi))

        psi[1,1,1] = 1/4
        psi[1,2,1] = 1/4
        psi[2,1,1] = -1/4
        psi[2,2,1] = -1/4
        self.hathaar2d[1,:,:] = fft.fft2(torch.view_as_complex(psi))

        psi[1,1,1] = 1/4
        psi[1,2,1] = -1/4
        psi[2,1,1] = -1/4
        psi[2,2,1] = 1/4
        self.hathaar2d[2,:,:] = fft.fft2(torch.view_as_complex(psi))

        # load masks for aperiodicity
        self.masks = maskns(J, M, N).unsqueeze(1).unsqueeze(1)  # (J, M, N)


    def filters_tensor(self):

        J = self.J
        M = self.M; N = self.N; L = self.L
        # load phi filters

        assert(self.M == self.N)
        if self.path is not None:
            matfilters = sio.loadmat(self.path+'matlab/filters/bumpsteerableg1_fft2d_N'
                                     + str(self.N) + '_J' + str(self.J) + '_L'
                                     + str(self.L) + '.mat')
    
            fftphi = matfilters['filt_fftphi'].astype(np.complex_)  # (M,N)
            hatphi = np.stack((np.real(fftphi), np.imag(fftphi)), axis=-1)
    
            fftpsi = matfilters['filt_fftpsi'].astype(np.complex_)  # (J,L2,M,N)
            # hatpsi = np.stack((np.real(fftpsi), np.imag(fftpsi)), axis=-1)
        else:
            Sihao_filters = FiltersSet(M=N, N=N, J=J, L=L).generate_bump_steerable()
            hatpsi_ = Sihao_filters['psi'] # (J,L,M,N)
            hatpsi_ = torch.cat((hatpsi_[...,None], hatpsi_[...,None]*0), dim=-1) # (J,L,M,N,2)
            hatpsi = torch.cat((hatpsi_, torch.flip(hatpsi_, (2,3))), dim=1).numpy() # (J,L2,M,N,2)
            fftpsi = hatpsi[...,0] + hatpsi[...,1]* 1.0j
            hatphi = Sihao_filters['phi']  # (M,N)
            hatphi = torch.cat((hatphi[...,None], hatphi[...,None]*0), dim=-1).numpy() # (M,N,2)



        J = self.J
        A = self.A
        A_prime = self.A_prime

        alphas = np.arange(A, dtype=np.float32)/(max(A,1))*np.pi*2
        alphas = np.exp(1j * alphas)


        filt = np.zeros((J, 2*L, A, self.M, self.N), dtype=np.complex_)
        filt_prime = np.zeros((J, 2*L, A_prime, self.M, self.N),
                              dtype=np.complex_)

        for alpha in range(A):
            for j in range(J):
                for theta in range(L):
                    psi_signal = fftpsi[j, theta, ...]
                    filt[j, theta, alpha, :, :] = alphas[alpha]*psi_signal
                    filt[j, L+theta, alpha, :, :] = np.conj(alphas[alpha]) \
                        * psi_signal

        alphas_prime = np.arange(A_prime, dtype=np.float32)/(max(A_prime,1))*np.pi*2
        alphas_prime = np.exp(1j * alphas_prime)

        for alpha in range(A_prime):
            for j in range(J):
                for theta in range(L):
                    psi_signal = fftpsi[j, theta, ...]
                    filt_prime[j, theta, alpha, :, :] = alphas_prime[alpha]*psi_signal
                    filt_prime[j, L+theta, alpha, :, :] = np.conj(alphas_prime[alpha])*psi_signal

        filters = np.stack((np.real(filt), np.imag(filt)), axis=-1)
        filters_prime = np.stack((np.real(filt_prime), np.imag(filt_prime)),
                                 axis=-1)

        self.hatphi = torch.FloatTensor(hatphi)  # (M,N,2)
        self.hatpsi = torch.FloatTensor(filters)
        self.hatpsi_prime = torch.FloatTensor(filters_prime)

        # load masks for aperiodicity
        self.masks = maskns(J, M, N).unsqueeze(1).unsqueeze(1)  # (J, M, N)



    def get_this_chunk(self, nb_chunks, chunk_id):
        # cut self.idx_wph into smaller pieces
        nb_cov = len(self.idx_wph['la1'])
        max_chunk = nb_cov // nb_chunks
        nb_cov_chunk = np.zeros(nb_chunks, dtype=np.int32)
        for idxc in range(nb_chunks):
            if idxc < nb_chunks-1:
                nb_cov_chunk[idxc] = int(max_chunk)
            else:
                nb_cov_chunk[idxc] = int(nb_cov - max_chunk*(nb_chunks-1))
                assert(nb_cov_chunk[idxc] > 0)

        this_wph = dict()
        offset = int(0)
        for idxc in range(nb_chunks):
            if idxc == chunk_id:
                this_wph['la1'] = self.idx_wph['la1'][offset:offset+nb_cov_chunk[idxc]]
                this_wph['la2'] = self.idx_wph['la2'][offset:offset+nb_cov_chunk[idxc]]
                this_wph['shifted'] = self.idx_wph['shifted'][offset:offset+nb_cov_chunk[idxc]]
            offset = offset + nb_cov_chunk[idxc]

        print('this chunk', chunk_id, ' size is ', len(this_wph['la1']), ' among ', nb_cov)

        return this_wph


    def to_shift(self, j1, j2, l1, l2):
        if self.shift == 'all':
            return True
        elif self.shift == 'same':
            return (j1 == j2) and (l1 ==l2)


    def compute_idx(self):
        L = self.L
        L2 = L*2
        J = self.J
        A = self.A
        A_prime = self.A_prime
        dj = self.delta_j
        dl = self.delta_l
        max_scale = int(np.log2(self.N))

        idx_la1 = []
        idx_la2 = []
        weights = []
        shifted = []
        nb_moments = 0

        for j1 in range(J):
            for j2 in range(j1, min(J,j1+1+dj)):
                for l1 in range(L):
                    for l2 in range(L):
                        for alpha1 in range(A):
                            for alpha2 in range(A_prime):
                                if self.to_shift(j1, j2, l1, l2):  # choose coeffs whith shits here
                                    idx_la1.append(A*L*j1+A*l1+alpha1)
                                    idx_la2.append(A*L*j2
                                                   + A*l2+alpha2)
                                    shifted.append(2)  # choose which shifts here
                                    nb_moments += int(self.factr_shift[-1])
                                else:
                                    idx_la1.append(A*L*j1+A*l1+alpha1)
                                    idx_la2.append(A_prime*L*j2
                                                   + A_prime*l2+alpha2)
                                    shifted.append(0)
                                    nb_moments += 1

        print('number of moments (without low-pass and harr): ', nb_moments)

        idx_wph = dict()
        idx_wph['la1'] = torch.tensor(idx_la1).type(torch.long)
        idx_wph['la2'] = torch.tensor(idx_la2).type(torch.long)
        idx_wph['shifted'] = torch.tensor(shifted).type(torch.long)

        return idx_wph

    def _type(self, _type):

        self.hatpsi = self.hatpsi.type(_type)
        self.hatpsi_prime = self.hatpsi_prime.type(_type)
        self.hatphi = self.hatphi.type(_type)
        if self.wavelets == 'morlet':
            self.hathaar2d = self.hathaar2d.type(_type)
        # self.filt_conv = self.filt_conv.type(_type)
        self.masks = self.masks.type(_type)
        self.masks_shift = self.masks_shift.type(_type)
        self.pad.padding_module.type(_type)

        return self

    def cuda(self):
        """
            Moves the parameters of the scattering to the GPU
        """
        if self.chunk_id < self.nb_chunks:
            self.this_wph['la1'] = self.this_wph['la1'].type(torch.cuda.LongTensor)
            self.this_wph['la2'] = self.this_wph['la2'].type(torch.cuda.LongTensor)

        self.hatpsi = self.hatpsi.cuda()
        self.hatphi = self.hatphi.cuda()
        if self.wavelets == 'morlet':
            self.hathaar2d = self.hathaar2d.cuda()
        self.masks = self.masks.cuda()
        self.masks_shift = self.masks_shift.cuda()
        self.pad.padding_module.cuda()

        return self #._type(torch.cuda.ComplexFloatTensor)

    def cpu(self):
        """
            Moves the parameters of the scattering to the CPU
        """
        return self._type(torch.FloatTensor)

    def forward(self, input):

        J = self.J
        M = self.M
        N = self.N
        A = self.A
        L = self.L
        L2 = 2*L
        phi = self.hatphi
        n = 0
        pad = self.pad
        wavelets = self.wavelets

        x_c = padc(input)  # add zeros to imag part -> (nb,M,N)
        hatx_c = fft.fft2(torch.view_as_complex(x_c)).type(torch.cfloat)  # fft2 -> (nb,M,N)

        if self.chunk_id < self.nb_chunks:
            nb = hatx_c.shape[0]
            hatpsi_la = self.hatpsi[:,:L,...]  # (J,L,A,M,N)
            nb_channels = self.this_wph['la1'].shape[0]
            t = 3 if wavelets == 'morlet' else 1 if wavelets == 'steer' else 0
            if self.chunk_id < self.nb_chunks-1:
                Sout = input.new(nb,nb_channels,M,N)
            else:
                Sout = input.new(nb,
                                  nb_channels+1+t,
                                  M,N)
            idxb = 0
            hatx_bc = hatx_c[idxb, :, :]  # (M,N)

            hatxpsi_bc = hatpsi_la * hatx_bc.view(1,1,1,M,N)  # (J,L2,A,M,N)
            xpsi_bc = fft.ifft2(hatxpsi_bc)
            xpsi_bc_ = torch.real(xpsi_bc).relu()
            xpsi_bc_ = xpsi_bc_ * self.masks
            xpsi_bc0 = self.subinitmean1(xpsi_bc_)
            xpsi_bc0_n = self.divinitstd1(xpsi_bc0)
            xpsi_bc0_ = xpsi_bc0_n.view(1, J*L*A, M, N)

            xpsi_bc_la1 = xpsi_bc0_[:,self.this_wph['la1'],...]   # (1,P_c,M,N)
            xpsi_bc_la2 = xpsi_bc0_[:,self.this_wph['la2'],...] # (1,P_c,M,N)

            x1 = torch.view_as_complex(padc(xpsi_bc_la1))
            x2 = torch.view_as_complex(padc(xpsi_bc_la2))
            hatconv_xpsi_bc = fft.fft2(x1) * torch.conj(fft.fft2(x2))
            conv_xpsi_bc = torch.real(fft.ifft2(hatconv_xpsi_bc))
            masks_shift = self.masks_shift[self.this_wph['shifted'],...].view(1,-1,M,N)
            corr_bc = conv_xpsi_bc * masks_shift

            Sout[idxb, 0:nb_channels,...] = corr_bc[0, ...]

            if self.chunk_id == self.nb_chunks-1:
                # ADD 1 channel for spatial phiJ
                # add l2 phiJ to last channel
                hatxphi_c = hatx_c * self.hatphi.view(1,M,N)  # (nb,nc,M,N,2)
                xphi_c = fft.fft2(hatxphi_c)
                # haar filters
                if wavelets == 'morlet':
                    for hid in range(3):
                        hatxpsih_c = hatx_c * self.hathaar2d[hid,:,:].view(1,M,N) # (nb,nc,M,N)
                        xpsih_c = fft.ifft2(hatxpsih_c)
                        xpsih_c = self.divinitstdH[hid](xpsih_c)
                        xpsih_c = xpsih_c * self.masks[0,...].view(1,M,N)
                        xpsih_mod = fft.fft2(torch.view_as_complex(padc(xpsih_c.abs())))
                        xpsih_mod2 = fft.ifft2(xpsih_mod * torch.conj(xpsih_mod))
                        xpsih_mod2 = torch.real(xpsih_mod2[0,...]) * self.masks_shift[-1,...]
                        Sout[idxb,-4+hid,...] = xpsih_mod2

                # submean from spatial M N
                xphi0_c = self.subinitmeanJ(xphi_c)
                xphi0_c = self.divinitstdJ(xphi0_c)
                xphi0_c = xphi0_c * self.masks[-1,...].view(1,M,N)
                xphi0_mod = fft.fft2(torch.view_as_complex(padc(xphi0_c.abs())))  # (nb,nc,M,N)
                xphi0_mod2 = fft.ifft2(xphi0_mod * torch.conj(xphi0_mod)) # (nb,nc,M,N)
                xphi0_mean = torch.real(xphi0_mod2) * self.masks_shift[-1,...].view(1,M,N)
                '''
                # low-high corr
                l_h_1 = padc(xpsi_bc0_).fft(2)
                l_h_2 = self.subinitmeanin(hatx_bc)
                l_h_2 = self.divinitstdin(l_h_2)
                l_h = mulcu(l_h_1, conjugate(l_h_2)).ifft(2)[...,0]
                l_h = l_h * self.masks_shift[1,...].view(1,1,M,N)
                Sout[idxb, idxc, nb_channels+t:-(1+t),...] = l_h[0,...]
                '''
                Sout[idxb, -1,...] = xphi0_mean[idxb, ...]

            Sout = Sout.view(nb, -1, M*N)[..., self.indices]
            Sout = Sout.view(-1)
            Sout = torch.cat((Sout, input.mean().view(1), input.std().view(1)))*1e-4


        return Sout

    def __call__(self, input):
        return self.forward(input)