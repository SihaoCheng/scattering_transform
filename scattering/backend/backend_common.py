import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReflectionPad2d
from torch.nn import ZeroPad2d
from torch.autograd import Function
import numpy as np



def pad_filters(f, J):
    M = f.size(-2)
    m = M//2
    d = f.dim()
    f_sp = f.ifft(2)  # (J,L,A,M,N,2) or (M,N,2)
    if d == 6:
        f_ = torch.zeros(f.size(0),
                         f.size(1),
                         f.size(2),
                         M + 2**J,
                         M + 2**J,
                         2)
    else:
        f_ = torch.zeros(M + 2**J,M + 2**J,2)
    f_[...,:m,:m,:] = f_sp[...,:m,:m,:]
    f_[...,:m,-m:,:] = f_sp[...,:m,-m:,:]
    f_[...,-m:,:m,:] = f_sp[...,-m:,:m,:]
    f_[...,-m:,-m:,:] = f_sp[...,-m:,-m:,:]

    return f_.fft(2)



def filters_tensor(J,L,M,N,A,A_prime):
    J2 = J
    matfilters = sio.loadmat('./matlab/filters/bumpsteerableg1_fft2d_N'
                             + str(N) + '_J' + str(J) + '_L'
                             + str(L) + '.mat')

    fftphi = matfilters['filt_fftphi'].astype(np.complex_)  # (M,N)
    hatphi = np.stack((np.real(fftphi), np.imag(fftphi)), axis=-1)

    fftpsi = matfilters['filt_fftpsi'].astype(np.complex_)  # (J,L2,M,N)
    hatpsi = np.stack((np.real(fftpsi), np.imag(fftpsi)), axis=-1)

    alphas = np.zeros(A) \
        + np.linspace(np.pi/(2*A),
                      (2*A-1)*np.pi/(2*A), A)*(1 - (A == 1))
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

    alphas_prime = np.zeros(A_prime) \
        + (np.linspace(np.pi/(2*A_prime),
                       (2*A_prime-1)*np.pi/(2*A_prime), A_prime)
           * (1 - (A_prime == 1)))

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

    hatphi = torch.FloatTensor(hatphi)  # (M,N,2)
    hatpsi = torch.FloatTensor(filters)
    hatpsi_prime = torch.FloatTensor(filters_prime)

    hatpsil2 = torch.FloatTensor(hatpsi) #[:,:L,...] # (J,L2,M,N,2)
    hatphil2 = torch.FloatTensor(hatphi) # (M,N,2)

    # shift second layer filters for roto-translation
    hatpsi2 = hatpsil2.clone().unsqueeze(2).repeat(1, 1, L2, 1, 1, 1)
    for dl in range(L2):
        hatpsi2[:, :, dl, :, :, :] = torch.cat((hatpsil2[:, dl:, ...], hatpsil2[:, :dl, ...]), dim=1)
    hatpsi2 = hatpsi2.view(J2*L2, L2, self.M, self.N, 2)

    # load angle filters
    filters1d = torch.load('./matlab/filters/ky_1d_H'+str(self.H)+'_L'+str(self.L)+'.pt')  # (H*L2/delta, L2, 2)

    return hatpsi, hatpsi_prime, hatphi, hatpsil2, hatphil2, filters1d

def shift(x):
    N = x.shape[0]
    x_ = torch.Tensor(x)
    x_ = torch.cat((x_[N//2:,:,:], x_[:N//2,:,:]), dim=0)
    xs = torch.cat((x_[:,N//2:,:], x_[:,:N//2,:]), dim=1)
    return xs

def dft(pos, res, nb_chunks, om):
    pts = pos.size(0)
    offset = 0

    for i in range(nb_chunks):
        pos_ = pos[offset:offset+pts//nb_chunks, :]
        nb_points = pos_.size(0)
        omegas = om.unsqueeze(0).repeat(nb_points, 1)
        pos_x = pos_[:, 0].unsqueeze(1).repeat(1, res)
        pos_y = pos_[:, 1].unsqueeze(1).repeat(1, res)

        prod_x = omegas*pos_x  # (nb_points, res)
        prod_y = omegas*pos_y  # (nb_points, res)

        prod = prod_x.unsqueeze(1).repeat(1, res, 1) + prod_y.unsqueeze(2).repeat(1, 1, res)  # (nb_points, res, res)

        exp = torch.stack((torch.cos(prod), -torch.sin(prod)), dim=-1).sum(0)  # (res, res, 2)

        if i==0:
            M = exp
        else:
            M = M + exp

        offset += pts//nb_chunks

    pos_ = pos[offset:, :]
    nb_points = pos_.size(0)
    omegas = om.unsqueeze(0).repeat(nb_points, 1)
    pos_x = pos_[:, 0].unsqueeze(1).repeat(1, res)
    pos_y = pos_[:, 1].unsqueeze(1).repeat(1, res)

    prod_x = omegas*pos_x  # (nb_points, res)
    prod_y = omegas*pos_y  # (nb_points, res)

    prod = prod_x.unsqueeze(1).repeat(1, res, 1) + prod_y.unsqueeze(2).repeat(1, 1, res)  # (nb_points, res, res)

    exp = torch.stack((torch.cos(prod), -torch.sin(prod)), dim=-1).sum(0)  # (res, res, 2)

    M = M + exp
    M = M.norm(dim=-1)**2

    return M




def maskns(J, M, N):
    m = torch.ones(J, M, N)
    for j in range(J):
        for x in range(M):
            for y in range(N):
                if (x<(2**j)//2 or y<(2**j)//2 \
                or x+1>M-(2**j)//2 or y+1>N-(2**j)//2):
                    m[j, x, y] = 0
    m = m.type(torch.float)
    m = m / m.sum(dim=(-1,-2), keepdim=True)
    m = m*M*N
    return m


def maskns_o2(J, M, N):
    m = torch.ones(2**J, M, N)
    for s in range(2**J):
        for x in range(M):
            for y in range(N):
                if (x<s-1 or y<s-1 \
                or x>M-s or y>N-s):
                    m[s, x, y] = 0
    m = m.type(torch.float)
    m = m / m.sum(dim=(-1,-2), keepdim=True)
    m = m*M*N
    return m

def masks_subsample_shift(J,M,N):
    m = torch.zeros(M,N).type(torch.float)
    m[0,0] = 1.
    angles = torch.arange(8).type(torch.float)
    angles = angles/8*2*np.pi
    for j in range(J):
        for theta in range(8):
            x = int(torch.round((2**j)*torch.cos(angles[theta])))
            y = int(torch.round((2**j)*torch.sin(angles[theta])))
            m[x,y] = 1.
    return m


def masks_shift_ps(J,M,N,dn):
    m = torch.zeros(J,M,N)
    for j in range(J):
        for x in range(M):
            for y in range(N):
                if (x <= (2**j)*dn or M-x <= (2**j)*dn) \
                and (y <= (2**j)*dn or N-y<= (2**j)*dn):
                    m[j,x,y] = 1.
    return m


def masks_subsample_shift2(J,M,N):
    m = torch.zeros(J,M,N).type(torch.float)
    m[:,0,0] = 1.
    angles = torch.arange(8).type(torch.float)
    angles = angles/8*2*np.pi
    for j in range(J):
        for theta in range(8):
            x = int(torch.round((2**j)*torch.cos(angles[theta])))
            y = int(torch.round((2**j)*torch.sin(angles[theta])))
            m[j,x,y] = 1.
    return m


def masks_subsample_shift3(J,M,N):
    m = torch.zeros(J,M,N).type(torch.float)
    m[:,0,0] = 1.
    angles = torch.arange(8).type(torch.float)
    angles = angles/8*2*np.pi
    for j in range(J):
        for theta in range(8):
            x = int(torch.round((2**j)*torch.cos(angles[theta])))
            y = int(torch.round((2**j)*torch.sin(angles[theta])))
            for j_ in range(j,J):
                m[j_,x,y] = 1.
    return m


def extract_shift3(x):
    M = x.size(-1)
    l = []
    for i in range(M):
        for j in range(M):
            if x[i,j] != 0:
                l.append(j + M*i)
    l = torch.tensor(l)
    return l





def masks_subsample_shift4(J,M,N):
    m = torch.zeros(J,M,N).type(torch.float)
    m[:,0,0] = 1.
    angles = torch.arange(8).type(torch.float)
    angles = angles/8*2*np.pi
    for j in range(J):
        for theta in range(8):
            x = int(torch.round((2**j)*torch.cos(angles[theta])))
            y = int(torch.round((2**j)*torch.sin(angles[theta])))
            for j_ in range(j+1):
                m[j_,x,y] = 1.
    return m


def tocplx(x):
    return torch.view_as_complex(x)

def toreal(x):
    return torch.view_as_real(x)


def iscomplex(input):
    return input.size(-1) == 2


def ones_like(z):
    re = torch.ones_like(z[..., 0])
    im = torch.zeros_like(z[..., 1])
    return torch.stack((re, im), dim=-1)

def real(z):
    z_ = z.clone()
    return z_[..., 0]


def imag(z):
    z_ = z.clone()
    return z_[..., 1]


def conjugate(z):
    z_copy = z.clone()
    z_copy[..., 1] = -z_copy[..., 1]
    return z_copy


def pows(z, max_k, dim=0):
    z_pows = [ones_like(z)]
    if max_k > 0:
        z_pows.append(z)
        z_acc = z
        for k in range(2, max_k + 1):
            z_acc = mul(z_acc, z)
            z_pows.append(z_acc)
    z_pows = torch.stack(z_pows, dim=dim)
    return z_pows


def log2_pows(z, max_pow_k, dim=0):
    z_pows = [ones_like(z)]
    if max_pow_k > 0:
        z_pows.append(z)
        z_acc = z
        for k in range(2, max_pow_k + 1):
            z_acc = mul(z_acc, z_acc)
            z_pows.append(z_acc)
    assert len(z_pows) == max_pow_k + 1
    z_pows = torch.stack(z_pows, dim=dim)
    return z_pows

def mul(z1_, z2_):
    z1 = z1_.clone()
    z2 = z2_.clone()
    zr = real(z1) * real(z2) - imag(z1) * imag(z2)
    zi = real(z1) * imag(z2) + imag(z1) * real(z2)
    z = z1.new(z1.size())
    z[...,0] = zr
    z[...,1] = zi
    return z

# substract spatial mean (complex valued input)
class SubInitSpatialMeanC(object):
    def __init__(self):
        self.minput = None

    def __call__(self, input):
        if self.minput is None:
            minput = input.clone().detach()
            minput = torch.mean(minput, -2, True)
            minput = torch.mean(minput, -3, True)
            self.minput = minput
#            print('sum of minput',self.minput.sum())

        output = input - self.minput
        return output

class SubInitSpatialMeanCrl(object):
    def __init__(self):
        self.minput = None

    def __call__(self, input):
        if self.minput is None:
            minput = input.clone().detach() 
            minput = torch.mean(minput, -2, True)
            minput = torch.mean(minput, -3, True)
            self.minput = minput
#            print('sum of minput',self.minput.sum())

        output = input - self.minput
        return output


# substract spatial mean (real valued input)
class SubInitSpatialMeanR(object):
    def __init__(self):
        self.minput = None

    def __call__(self, input):
        if self.minput is None:
            minput = input.clone().detach()
            minput = torch.mean(minput, -1, True)
            minput = torch.mean(minput, -2, True)
            self.minput = minput
#            print('sum of minput',self.minput.sum())

        output = input - self.minput
        return output

class SubInitMeanIso(object):
    def __init__(self):
        self.minput = None

    def __call__(self, input):
        if self.minput is None:
            minput = input.clone().detach()  # input size:(J,Q,K,M,N,2)
            minput = torch.mean(minput, -2, True)
            minput = torch.mean(minput, -3, True)
            minput[:, 1:, ...] = 0
            self.minput = minput
#            print('sum of minput', self.minput.sum())
            # print('minput shape is', self.minput.shape)
        output = input - self.minput
        return output

class DivInitStd(object):
    def __init__(self,stdcut=1e-6):
        self.stdinput = None
        self.eps = stdcut
#        print('DivInitStd:stdcut',stdcut)

    def __call__(self, input):
        if self.stdinput is None:
            stdinput = input.clone().detach()  # input size:(J,Q,K,M,N,2)
            m = torch.mean(torch.mean(stdinput, -2, True), -3, True)
            stdinput = stdinput - m
            d = input.shape[-2]*input.shape[-3]
            stdinput = torch.norm(stdinput, dim=-1, keepdim=True)
            stdinput = torch.norm(stdinput, dim=(-3,-2), keepdim=True)
            self.stdinput = stdinput  / np.sqrt(d)
            self.stdinput = self.stdinput + self.eps
#            print('stdinput max,min:',self.stdinput.max(),self.stdinput.min())

        output = input/self.stdinput
#        print(self.stdinput)
        return output

class DivInitStdrl(object):
    def __init__(self,stdcut=1e-6):
        self.stdinput = None
        self.eps = stdcut
#        print('DivInitStd:stdcut',stdcut)

    def __call__(self, input):
        if self.stdinput is None:
            stdinput = input.clone().detach()  # input size:(J,Q,K,M,N,2)
            m = torch.mean(torch.mean(stdinput, -2, True), -3, True)
            stdinput = stdinput - m
            d = input.shape[-2]*input.shape[-3]
            #stdinput = torch.norm(stdinput, dim=-1, keepdim=True)
            stdinput = torch.norm(stdinput, dim=-2, keepdim=True).norm(dim=-3, keepdim=True)
            self.stdinput = stdinput  / np.sqrt(d)
            self.stdinput = self.stdinput + self.eps
#            print('stdinput max,min:',self.stdinput.max(),self.stdinput.min())

        output = input/self.stdinput
#        print(self.stdinput)
        return output

class DivInitStdR(object):
    def __init__(self,stdcut=1e-6):
        self.stdinput = None
        self.eps = stdcut
#        print('DivInitStd:stdcut',stdcut)

    def __call__(self, input):
        if self.stdinput is None:
            stdinput = input.clone().detach()  # input size:(...,M,N)
            m = torch.mean(torch.mean(stdinput, -1, True), -2, True)
            stdinput = stdinput - m
            d = input.shape[-1]*input.shape[-2]
            stdinput = torch.norm(stdinput, dim=(-2,-1), keepdim=True)
            self.stdinput = stdinput  / np.sqrt(d)
            self.stdinput = self.stdinput + self.eps
#            print('stdinput max,min:',self.stdinput.max(),self.stdinput.min())

        output = input/self.stdinput
        return output

class DivInitStdRot(object):
    def __init__(self,stdcut=0):
        self.stdinput = None
        self.eps = stdcut
#        print('DivInitStd:stdcut',stdcut)

    def __call__(self, input):
        if self.stdinput is None:
            stdinput = input.clone().detach()  # input size:(1,P_c,M,N,2)
            m = torch.mean(torch.mean(stdinput, -2, True), -3, True)
            stdinput = stdinput - m
            d = input.shape[-2]*input.shape[-3]
            stdinput = torch.norm(stdinput, dim=-1, keepdim=True)
            stdinput = torch.norm(stdinput, dim=(-2, -3), keepdim=True)
            self.stdinput = stdinput  / np.sqrt(d)
            self.stdinput = self.stdinput + self.eps
#            print('stdinput max,min:',self.stdinput.max(),self.stdinput.min())

        output = input/self.stdinput
        return output



class DivInitMax(object):
    def __init__(self):
        self.max = None

    def __call__(self, input):
        if self.max is None:
            maxinput = input.clone().detach()  # input size:(1,P_c,M,N)
            maxinput = torch.max(maxinput, dim=-1, keepdim=True)[0]
            maxinput = torch.max(maxinput, dim=-2, keepdim=True)[0]
            self.max = maxinput
        output = input/self.max

        return output


class DivInitMean(object):
    def __init__(self):
        self.mean = None

    def __call__(self, input):
        if self.mean is None:
            if input.size(-1) > 2:
                meaninput = input.clone().detach()  # input size:(1,P_c,M,N)
                meaninput = torch.mean(meaninput, dim=-1, keepdim=True)[0]
                meaninput = torch.mean(meaninput, dim=-2, keepdim=True)[0]
                self.mean = meaninput+1e-6
            else:
                meaninput = input.clone().detach()  # input size:(1,P_c,M,N,2)
                meaninput = torch.mean(meaninput, dim=-2, keepdim=True)
                meaninput = torch.mean(meaninput, dim=-3, keepdim=True)
                self.mean = meaninput

        output = input/self.mean

        return output


class DivInitStdQ0(object):
    def __init__(self):
        self.stdinput = None
        self.eps = 1e-16

    def __call__(self, input):
        if self.stdinput is None:
            stdinput = input.clone().detach()  # input size:(J,Q,K,M,N,2)
            stdinput = stdinput[:, :1, ...]  # size:(J,1,K,M,N,2)
            d = input.shape[-2]*input.shape[-3]
            stdinput = torch.norm(stdinput, dim=-1, keepdim=True)
            stdinput = torch.norm(stdinput, dim=(-2, -3), keepdim=True) / np.sqrt(d)
            self.stdinput = stdinput + self.eps
#            print('stdinput max,min:',self.stdinput.max(),self.stdinput.min())
        output = input/self.stdinput
        return output

# substract spatial mean (complex valued input), average over ell
class SubInitSpatialMeanCL(object):
    def __init__(self):
        self.minput = None

    def __call__(self, input): # input: (J,L2,M,N,K,2)
        if self.minput is None:
            minput = input.clone().detach()
            minput = torch.mean(minput, dim=1, keepdim=True)
            minput = torch.mean(minput, dim=2, keepdim=True)
            minput = torch.mean(minput, dim=3, keepdim=True)
            self.minput = minput # .expand_as(input)
#            print('minput size',self.minput.shape)
#            print('sum of minput',self.minput.sum())

        output = input - self.minput
        return output

# divide by std, average over ell
class DivInitStdL(object):
    def __init__(self):
        self.stdinput = None

    def __call__(self, input): # input: (J,L2,M,N,K,2)
        if self.stdinput is None:
            stdinput = input.clone().detach()
            #dl = input.shape[1]*input.shape[2]*input.shape[3]
            stdinput = torch.norm(stdinput, dim=-1, keepdim=True) # (J,L2,M,N,K,1)
            stdinput = torch.mul(stdinput,stdinput)
            stdinput = torch.mean(stdinput, dim=1, keepdim=True)
            stdinput = torch.mean(stdinput, dim=2, keepdim=True)
            stdinput = torch.mean(stdinput, dim=3, keepdim=True)
            self.stdinput = torch.sqrt(stdinput) # .expand_as(input) #  / dl)
#            print('stdinput size',self.stdinput.shape)
#            print('stdinput max,min:',self.stdinput.max(),self.stdinput.min())

        output = input/self.stdinput
        return output

class SubInitSpatialMeanCinFFT(object):
    def __init__(self):
        self.minput = None

    def __call__(self, input):
        if self.minput is None:
            minput = input.clone().detach()
            minput = input[...,0,0,:] # zero-freq. value torch.mean(minput, -2, True)
            #minput = torch.mean(minput, -3, True)
            self.minput = minput
#            print('sum of minput',self.minput.sum())

        output = input
        output[...,0,0,:] = input[...,0,0,:] - self.minput
        return output

class SubsampleFourier(object):
    """
        Subsampling of a 2D image performed in the Fourier domain
        Subsampling in the spatial domain amounts to periodization
        in the Fourier domain, hence the formula.
        Parameters
        ----------
        x : tensor_like
            input tensor with at least 5 dimensions, the last being the real
             and imaginary parts.
            Ideally, the last dimension should be a power of 2 to avoid errors.
        k : int
            integer such that x is subsampled by 2**k along the spatial variables.
        Returns
        -------
        res : tensor_like
            tensor such that its fourier transform is the Fourier
            transform of a subsampled version of x, i.e. in
            FFT^{-1}(res)[u1, u2] = FFT^{-1}(x)[u1 * (2**k), u2 * (2**k)]
    """
    def __call__(self, input, k):
        if input.ndim == 5:
            out = input.new(input.size(0), input.size(1), input.size(2) // k, input.size(3) // k, 2)


            y = input.view(input.size(0), input.size(1),
                           input.size(2)//out.size(2), out.size(2),
                           input.size(3)//out.size(3), out.size(3),
                           2)

            out = y.mean(4, keepdim=False).mean(2, keepdim=False)
        elif input.ndim==4:
            out = input.new(input.size(0), input.size(1) // k, input.size(2) // k, 2)


            y = input.view(input.size(0),
                           input.size(1)//out.size(1), out.size(1),
                           input.size(2)//out.size(2), out.size(2),
                           2)

            out = y.mean(3, keepdim=False).mean(1, keepdim=False)
        else:  #ndim=3
            out = input.new(input.size(0) // k, input.size(1) // k, 2)


            y = input.view(
                           input.size(0)//out.size(0), out.size(0),
                           input.size(1)//out.size(1), out.size(1),
                           2)

            out = y.mean(2, keepdim=False).mean(0, keepdim=False)

        return out

class SubsampleFourier2(object):

    def __call__(self, input, k):

        k_ = 2*k
        # ndim = 5
        out = input.new(input.size(0), input.size(1), input.size(2) // k_, input.size(3) // k_, 2)

        y = input.view(input.size(0), input.size(1),
                           input.size(2)//out.size(2), out.size(2),
                           input.size(3)//out.size(3), out.size(3),
                           2)
        s = [0,-1]
        out = y[:,:,s,:,s,:,:]

        return out

class SubInitMean(object):
    def __init__(self, dim):
        self.dim = dim # use the last "dim" dimensions to compute the mean
        self.minput = None

    def __call__(self, input):
        if self.minput is None:
            minput = input.clone().detach()
            #print('subinitmean:input',input.shape)
            for d in range(self.dim):
                minput = torch.mean(minput, -1)
            for d in range(self.dim):
                minput = minput.unsqueeze(-1)
            #print('subinitmean:minput',minput.shape)
            minput.expand_as(input)
            self.minput = minput

        #print('subinitmean:minput sum',self.minput.sum())
        output = input - self.minput
        return output

class Pad(object):
    def __init__(self, pad_size, pre_pad=False, pad_mode='zero'):
        """
            Padding which allows to simultaneously pad in a reflection fashion
            and map to complex.
            Parameters
            ----------
            pad_size : int
                size of padding to apply.
            pre_pad : boolean
                if set to true, then there is no padding, one simply adds the imaginarty part.
        """
        self.pre_pad = pre_pad
        if pad_mode == 'Reflect':
#            print('use reflect pad')
            self.padding_module = ReflectionPad2d(pad_size)
        else:
#            print('use zero pad')
            self.padding_module = ZeroPad2d(pad_size)

    def __call__(self, input):
        if(self.pre_pad):
            output = input.new_zeros(input.size(0), input.size(1), input.size(2), input.size(3), 2)
            output.narrow(output.ndimension()-1, 0, 1)[:] = input
        else:
            out_ = self.padding_module(input)
            output = input.new_zeros(*(out_.size() + (2,)))
            output.select(4, 0)[:] = out_

        return output



def padc(x):
    x_ = x.clone()
    return torch.stack((x_, torch.zeros_like(x_)), dim=-1)


def rot_mat(K):
    M = toch.eye(K).unsqueeze(0)
    M_ = toch.eye(K)
    for k in range(1, K):
        M_ = torch.cat((M_[:, 1:], M_[:, :1]), dim=1)
        M = torch.stack((M, M_), dim=0)
    return M


def rot_vec(v, M):
    """
        v: Tensor of size (1, J*L2*(dn+1), M, N, 2)
    """
    K = v.shape[0]
    v_ = v.unsqueeze(0).unsqueeze(0).repeat(K,1,1)
    z = bmm(v_, M).view(K, K)
    return z




def unpad(in_):
    """
        Slices the input tensor at indices between 1::-1
        Parameters
        ----------
        in_ : tensor_like
            input tensor
        Returns
        -------
        in_[..., 1:-1, 1:-1]
    """
    return in_[..., 1:-1, 1:-1]

class Modulus(object):
    """
        This class implements a modulus transform for complex numbers.
        Usage
        -----
        modulus = Modulus()
        x_mod = modulus(x)
        Parameters
        ---------
        x: input tensor, with last dimension = 2 for complex numbers
        Returns
        -------
        output: a tensor with imaginary part set to 0, real part set equal to
        the modulus of x.
    """
    def __call__(self, input):

        norm = input.norm(p=2, dim=-1, keepdim=True)
        return torch.cat([norm, torch.zeros_like(norm)], -1)


def modulus(z):
    z_mod = z.norm(p=2, dim=-1)

    # if z.requires_grad:
    #     # z_mod_mask.register_hook(HookDetectNan("z_mod_mask in modulus"))
    #     z_mod.register_hook(HookDetectNan("z_mod in modulus"))
    #     z.register_hook(HookDetectNan("z in modulus"))

    return z_mod


def fft(input, direction='C2C', inverse=False):
    """
        Interface with torch FFT routines for 2D signals.
        Example
        -------
        x = torch.randn(128, 32, 32, 2)
        x_fft = fft(x, inverse=True)
        Parameters
        ----------
        input : tensor
            complex input for the FFT
        direction : string
            'C2R' for complex to real, 'C2C' for complex to complex
        inverse : bool
            True for computing the inverse FFT.
            NB : if direction is equal to 'C2R', then the transform
            is automatically inverse.
    """
    if direction == 'C2R':
        inverse = True

    if not iscomplex(input):
        raise(TypeError('The input should be complex (e.g. last dimension is 2)'))

    if (not input.is_contiguous()):
        raise (RuntimeError('Tensors must be contiguous!'))

    if direction == 'C2R':
        output = torch.irfft(input, 2, normalized=False, onesided=False)*input.size(-2)*input.size(-3)
    elif direction == 'C2C':
        if inverse:
            output = torch.ifft(input, 2, normalized=False)*input.size(-2)*input.size(-3)
        else:
            output = torch.fft(input, 2, normalized=False)

    return output



class PhaseHarmonics2(Function):
    @staticmethod
    def forward(ctx, z, k):
        z = z.detach()
        x, y = real(z), imag(z)
        r = z.norm(p=2, dim=-1)
        theta = torch.atan2(y, x)
        ktheta = k * theta
        eiktheta = torch.stack((torch.cos(ktheta), torch.sin(ktheta)), dim=-1)
        ctx.save_for_backward(x, y, r, k)
        return r.unsqueeze(-1)*eiktheta

    @staticmethod
    def backward(ctx, grad_output):
        x, y, r, k = ctx.saved_tensors
        theta = torch.atan2(y, x)
        ktheta = k * theta
        cosktheta = torch.cos(ktheta)
        sinktheta = torch.sin(ktheta)
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        df1dx = costheta*cosktheta + k*sintheta*sinktheta
        df2dx = costheta*sinktheta - k*sintheta*cosktheta
        df1dy = sintheta*cosktheta - k*costheta*sinktheta
        df2dy = sintheta*sinktheta + k*costheta*cosktheta

        dx1 = df1dx*grad_output[...,0] + df2dx*grad_output[...,1]
        dx2 = df1dy*grad_output[...,0] + df2dy*grad_output[...,1]

        return torch.stack((dx1, dx2), -1), k # dummy gradient torch.zeros_like(k)

class PhaseHarmonicsIso(Function):
    # z.size(): (J,L2,M,N,1,2)
    # k.size(): (K)
    @staticmethod
    def forward(ctx, z, k):
        z = z.detach()
        x, y = real(z), imag(z)
        r = z.norm(p=2, dim=-1)  # (J, L2, M, N, 1)
        theta = torch.atan2(y, x)  # (J, L2, M, N, 1)
#        print(theta.size(), k.size())
        ktheta = k * theta  # (J, L2, M, N, K)
        eiktheta = torch.stack((torch.cos(ktheta), torch.sin(ktheta)), dim=-1)
        # eiktheta.size(): (J, L2, M, N, K, 2)
        ctx.save_for_backward(x, y, r, k)
        return r.unsqueeze(-1)*eiktheta

    @staticmethod
    def backward(ctx, grad_output):
        x, y, r, k = ctx.saved_tensors
        theta = torch.atan2(y, x)
        ktheta = k * theta
        cosktheta = torch.cos(ktheta)
        sinktheta = torch.sin(ktheta)
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        df1dx = costheta*cosktheta + k*sintheta*sinktheta
        df2dx = costheta*sinktheta - k*sintheta*cosktheta
        df1dy = sintheta*cosktheta - k*costheta*sinktheta
        df2dy = sintheta*sinktheta + k*costheta*cosktheta

        dx1 = df1dx*grad_output[..., 0] + df2dx*grad_output[..., 1]
        dx2 = df1dy*grad_output[..., 0] + df2dy*grad_output[..., 1]

        return torch.stack((dx1, dx2), -1), torch.zeros_like(k)

#phase_exp = PhaseHarmonics2.apply

# rest



class StablePhase(Function):
    @staticmethod
    def forward(ctx, z):
        z = z.detach()
        x, y = real(z), imag(z)
        r = z.norm(p=2, dim=-1)

        # NaN positions
        eps = 1e-32
        mask_real_neg = (torch.abs(y) <= eps) * (x <= 0)
        mask_zero = r <= eps

        x_tilde = r + x
        # theta = torch.atan(y / x_tilde) * 2
        theta = torch.atan2(y, x)

        # relace NaNs
        theta.masked_fill_(mask_real_neg, np.pi)
        theta.masked_fill_(mask_zero, 0.)

        # ctx.save_for_backward(x.detach(), y.detach(), r.detach())
        ctx.save_for_backward(x, y, r, mask_real_neg, mask_zero)
        return theta

    @staticmethod
    def backward(ctx, grad_output):
        x, y, r, mask_real_neg, mask_zero = ctx.saved_tensors

        # some intermediate variables
        x_tilde = r + x
        e = x_tilde ** 2 + y ** 2

        # derivative with respect to the real part
        dtdx = - y * x_tilde * 2 / (r * e)
        mask_real_neg_bis = (torch.abs(y) == 0) * (x <= 0)
        dtdx.masked_fill_(mask_real_neg, 0)
        dtdx.masked_fill_(mask_zero, 0)

        # derivative with respect to the imaginary part
        dtdy = x * x_tilde * 2 / (r * e)
        dtdy[mask_real_neg] = -1 / r[mask_real_neg]
        # dtdy.masked_fill_(mask, 0)
        dtdy.masked_fill_(mask_zero, 0)

        dtdz = grad_output.unsqueeze(-1) * torch.stack((dtdx, dtdy), dim=-1)
        return dtdz

phase = StablePhase.apply


class PhaseHarmonic(nn.Module):
    def __init__(self, check_for_nan=False):
        super(PhaseHarmonic, self).__init__()
        self.check_for_nan = check_for_nan



class StablePhaseExp(Function):
    @staticmethod
    def forward(ctx, z, r):
        eitheta = z / r
        eitheta.masked_fill_(r == 0, 0)

        ctx.save_for_backward(eitheta, r)
        return eitheta

    @staticmethod
    def backward(ctx, grad_output):
        eitheta, r = ctx.saved_tensors

        dldz = grad_output / r
        dldz.masked_fill_(r == 0, 0)

        dldr = - eitheta * grad_output / r
        dldr.masked_fill_(r == 0, 0)
        dldr = dldr.sum(dim=-1).unsqueeze(-1)

        return dldz, dldr


phaseexp = StablePhaseExp.apply


# periodic shift in 2d

class PeriodicShift2D(nn.Module):
    def __init__(self, M,N,shift1,shift2):
        super(PeriodicShift2D, self).__init__()
        self.M = M
        self.N = N
        self.shift1 = shift1 % M # [0,M-1]
        self.shift2 = shift2 % N # [0,N-1]

    def forward(self, input):
        # input dim is (1,P_c,M,N,2)
        # per. shift along M and N dim by shift1 and shift2
        #M = input.shape[2]
        #N = input.shape[3]
        M = self.M
        N = self.N
        shift1 = self.shift1
        shift2 = self.shift2

        #blk11 = [[0,0],[shift1-1,shift2-1]]
        #blk22 = [[shift1,shift2],[M-1,N-1]]
        #blk12 = [[shift1,0],[M-1,shift2-1]]
        #blk21 = [[0,shift2],[shift1-1,N-1]]
        output = input.clone()
        output[:,:,0:M-shift1,0:N-shift2,:] = input[:,:,shift1:M,shift2:N,:]
        output[:,:,0:M-shift1,N-shift2:N,:] = input[:,:,shift1:M,0:shift2,:]
        output[:,:,M-shift1:M,0:N-shift2,:] = input[:,:,0:shift1,shift2:N,:]
        output[:,:,M-shift1:M,N-shift2:N,:] = input[:,:,0:shift1,0:shift2,:]

        return output


def complex_mul(a, b):
    ar = a[..., 0]
    br = b[..., 0]
    ai = a[..., 1]
    bi = b[..., 1]
    real = ar*br - ai*bi
    imag = ar*bi + ai*br

    return torch.stack((real, imag), dim=-1)



def shift_filt(im_, u):
    # u: (J, L2, 2)
    # im_: (J, l2, M, N, 2)
    size = im_.size(-2)
    u = u.type(torch.float)
    map = torch.arange(size, dtype = torch.float).repeat(tuple(u.size()[:2])+(1,))  # (J, L2, N)
    z = torch.matmul(map.unsqueeze(-1), u.unsqueeze(-2))  # (J, L2, N, 1), (J, L2, 1, 2)->(J, L2, N, 2)
    sp = z[..., 0].unsqueeze(-1).repeat(1,1,1,size) + z[..., 1].unsqueeze(-2).repeat(1,1,size,1)  # (J, L2, N, N)
    del(z)
    # compute e^(-iw.u0)
    fft_shift = torch.stack((torch.cos(2*np.pi*sp/size), torch.sin(2*np.pi*sp/size)), dim=-1)  # (J, L2, N, N, 2)
    im_shift_fft = mul(fft_shift, im_)  # (J, L2, N, N, 2)
    del(fft_shift); del(sp)
    return im_shift_fft



def shift2(im_, u, devid):
    # u: (1, P_c ,m, 2)
    # im_: (1, P_c, N, N)
    size = im_.size(-2)
    u = u.type(torch.cuda.FloatTensor)
    im = torch.stack((im_, torch.cuda.FloatTensor(im_.size()).fill_(0)), dim=-1)  # (1, P_c, N, N, 2)
    im_fft = torch.fft(im, 2)  # (1, P_c, N, N, 2)
    map = torch.arange(size, dtype = torch.float, device = devid).repeat(tuple(u.size()[:3])+(1,))  # (1, P_c, m, N)
    z = torch.matmul(map.unsqueeze(-1), u.unsqueeze(-2))  # (1, P_c, m, N, 1), (1, P_c, m, 1, 2)->(1, P_c, m, N, 2)
    sp = z[..., 0].unsqueeze(-1).repeat(1,1,1,1,size) + z[..., 1].unsqueeze(-2).repeat(1,1,1,size,1)  # (1, P_c, m, N, N)
    del(z)
    # compute e^(-iw.u0)
    fft_shift = torch.stack((torch.cos(2*np.pi*sp/size), torch.sin(2*np.pi*sp/size)), dim=-1)  # (1, P_c, m, N, N, 2)
    im_shift_fft = mul(fft_shift, im_fft.unsqueeze(2).repeat(1,1,u.size(2), 1, 1, 1))  # (1, P_c, m, N, N, 2)
    del(fft_shift); del(sp); del(im_fft)
    im_shift = torch.ifft(im_shift_fft, 2)[..., 0]  # (1, P_c, m, N, N)
    return im_shift


def unshift2(ims_, u, devid):
    size = ims_.size(-2)
    u = u.type(torch.cuda.FloatTensor).to(devid)
    ims = torch.stack((ims_, torch.zeros(ims_.size()).type(torch.cuda.FloatTensor)), dim=-1)  # (1, P_c, N, N, 2)
    ims_fft = torch.fft(ims, 2)  # (1, P_c, m, N, N, 2)
    map = torch.arange(size, dtype = torch.float, device = devid).repeat(tuple(u.size()[:3])+(1,)).to(devid)  # (1, P_c, m, N)
    u_back = -u  # (1, P_c, m, 2)
    z = torch.matmul(map.unsqueeze(-1), u_back.unsqueeze(-2))  # (1, P_c, m, N, 1), (1, P_c, m, 1, 2)->(1, P_c, m, N, 2)
    sp = z[..., 0].unsqueeze(-1).repeat(1,1,1,1,size) + z[..., 1].unsqueeze(-2).repeat(1,1,1,size,1)  # (1, P_c, m, N, N)
    del(z)
    # compute e^(-iw.u0)
    fft_shift = torch.stack((torch.cos(2*np.pi*sp/size), torch.sin(2*np.pi*sp/size)), dim=-1)  # (1, P_c, m, N, N, 2)
    im_shift_fft = mul(ims_fft, fft_shift)  # (1, P_c, m, N, N, 2)
    del(fft_shift); del(sp); del(ims_fft)
    im_shift = torch.ifft(im_shift_fft, 2)[..., 0]  # (1, P_c, m, N, N)
    return im_shift

def indices_unpad(im, indices, pad, devid):
    size = im.size(-2)
    indices = indices.view(tuple(im.size()[:2])+(-1,))
    indices_col = (indices % (size+2*pad)).type(torch.cuda.LongTensor).to(devid)
    indices_col = (indices_col - pad)%size
    indices_row = ((indices )//(size+2*pad)).type(torch.cuda.LongTensor).to(devid)
    indices_row = (indices_row - pad)%size

    indices = size*indices_row + indices_col
    return indices, indices_row, indices_col


def local_max_indices(im, ks, nb_centers, devid):
    #  im: (1, P_c, N, N)
    size = im.size(-1)
    mp1 = torch.nn.MaxPool2d(2*ks+1, stride=torch.Size([1,1]), return_indices=True)

    im_pad1 = F.pad(im, (ks, ks, ks, ks), mode='circular')

    maxed1 = mp1(im_pad1)

    maxed1_ = indices_unpad(im, maxed1[1], ks, devid)

    z = torch.arange(size**2, dtype = torch.float, device = devid).unsqueeze(0).unsqueeze(0)
    z = z.repeat(im.size(0), im.size(1), 1)

    eq = torch.eq(maxed1_[0], z).type(torch.cuda.FloatTensor).to(devid)  # (1, P_c, N*N)

    '''
    imx = maxed1[0].view(im.size(0), im.size(1), -1)*eq  # (1, P_c, N*N)
    zero_count = (imx == 0).sum(dim=-1).max()
    top = torch.topk(imx, min(nb_centers, size*size-zero_count))[1]  # (1, P_c, number_of_centers)
    '''
    imx = maxed1[0].view(im.size(0), im.size(1), -1)*eq - (1-eq)  # (1, P_c, N*N)
    imx = torch.cat((imx, torch.zeros(im.size(0), im.size(1),
                                       nb_centers).type(torch.cuda.FloatTensor).to(devid)), -1)  # (1, P_c, N*N+nb_centers)
    null_count = (imx == -1).sum(dim=-1).min()
    top = torch.topk(imx, min(nb_centers, size*size-null_count))[1]  # (1, P_c, number_of_centers)
    top = top*top.lt(size**2).type(torch.cuda.FloatTensor).to(devid) \
        +(size**3)*top.ge(size**2).type(torch.cuda.FloatTensor).to(devid)


    del(eq); del(maxed1); #del(maxed2)
#    return top, torch.stack((top//size, top%size), dim=-1)
    return top, torch.stack((top//size, top%size), dim=-1).clone().requires_grad_(False)



class loc_maxs(object):
    def __init__(self, ks, nb_centers, devid):
        self.centers = None
        self.ks = ks
        self.nb_centers = nb_centers
        self.devid = devid

    def __call__(self, input):
        if self.centers is None:
            centers = local_max_indices(input, self.ks, self.nb_centers, self.devid)[1]
        return centers


#def local_max_indices(im, p):
#    return p.repeat(1, im.size(1), 1, 1)


def orth_phase(im2, loc, devid):
    #im2 (1, P_c, N, N, 2)
    if loc.size(2) == 1:
#        print('ploush')
        phase = torch.atan2(im2[...,1], im2[...,0])  # (1, P_c, N, N)
        shifted_phase = shift2(phase, loc, devid)  # (1, P_c, m, N, N)
        t_s_phase = torch.transpose(shifted_phase, -1, -2)  # (1, P_c, m, N, N)
        del(shifted_phase)
        t_s_phase = torch.flip(t_s_phase.unsqueeze(-2), [-2,-3]).squeeze().unsqueeze(0).unsqueeze(2)  # (1, P_c, m, N, N)
        orth_ph = unshift2(t_s_phase, loc, devid)  # (1, P_c, m, N, N)
        del(t_s_phase)
    else:
        phase = torch.atan2(im2[...,1], im2[...,0])  # (1, P_c, N, N)
        shifted_phase = shift2(phase, loc, devid)  # (1, P_c, m, N, N)
        t_s_phase = torch.transpose(shifted_phase, -1, -2)  # (1, P_c, m, N, N)
        del(shifted_phase)
        t_s_phase = torch.flip(t_s_phase.unsqueeze(-2), [-2,-3]).squeeze().unsqueeze(0)  # (1, P_c, m, N, N)
        orth_ph = unshift2(t_s_phase, loc, devid)  # (1, P_c, m, N, N)
        del(t_s_phase)
    return phase, orth_ph


def orth_phase2(im2, loc, devid):
    """
    Given a batch of images and a tensor of local maxima, this function returns
    a tuple consisting of the phase and the orthogonal phase centered at the local
    minima.
    """
    #im2 (1, P_c, N, N, 2)
    if loc.size(2) == 1:
        size = im2.size(-2)
        phase = torch.atan2(im2[...,1], im2[...,0])  # (1, P_c, N, N)
        #shifted_phase = shift2(phase, loc, devid)  # (1, P_c, m, N, N)

        z = torch.arange(-size,size).repeat(tuple(loc.size()[:3])+(1,)).type(torch.cuda.FloatTensor).to(devid)  # (1, P_c, m, N)
        z1 = z.unsqueeze(-1).repeat(1, 1, 1, 1, 2*size)  # (1, P_c, m, N, N)
        z2 = z.unsqueeze(-2).repeat(1, 1, 1, 2*size, 1)  # (1, P_c, m, N, N)
        z = z1**2 + z2**2
        del z1; del z2
        z = unshift2(z, -torch.cuda.FloatTensor([size,size]).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1,loc.size(1),loc.size(2),1),
                     devid).to(devid)
        #z[:,:,:,0,0] = 1
        z = z.unsqueeze(-1)


        c = torch.cuda.FloatTensor([size/2,size/2]).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1,loc.size(1),loc.size(2),1).to(devid)
        shifted_phase = shift2(phase, loc+c, devid)
        cplx_phase = torch.stack((torch.cos(shifted_phase), torch.sin(shifted_phase)), dim=-1)

        cplx_phase_p1 = torch.cat((cplx_phase, cplx_phase.unsqueeze(-2).flip(-2,-3).squeeze().unsqueeze(0).unsqueeze(0)),-2)
        cplx_phase_p = torch.cat((cplx_phase_p1, cplx_phase_p1.unsqueeze(-3).flip(-3,-4).squeeze().unsqueeze(0).unsqueeze(0)),-3)
        del cplx_phase; del cplx_phase_p1
        cpi = complex_mul(torch.ifft(z*torch.fft(cplx_phase_p,2),2),conjugate(cplx_phase_p))[...,1]
        del cplx_phase_p
        z_ = z.clone()
        del z
    #    z_[:,:,:,size,size,:] = z_[:,:,:,size,size,:]+1
        lin_sp_c_dx = torch.ifft(torch.fft(torch.stack((cpi, 0*cpi.clone()),dim=-1),2)/z_, 2)[...,:size,:size,0]
        del cpi;
        lin_sp_c_dx_0 = lin_sp_c_dx - lin_sp_c_dx.clone()[:,:,:,size//2,size//2].unsqueeze(-1).unsqueeze(-1)
        lin_sp_c_dx_ = unshift2(lin_sp_c_dx_0, c, devid)
        del lin_sp_c_dx; del lin_sp_c_dx_0
        t_s_phase = torch.transpose(lin_sp_c_dx_, -1, -2)  # (1, P_c, m, N, N)
        del(shifted_phase)
        t_s_phase = torch.flip(t_s_phase.unsqueeze(-2), [-2,-3]).squeeze().unsqueeze(0).unsqueeze(0)  # (1, P_c, m, N, N)
        orth_ph = unshift2(t_s_phase, loc, devid)  # (1, P_c, m, N, N)
        phase_ = unshift2(lin_sp_c_dx_, loc, devid)
        del(t_s_phase); del lin_sp_c_dx_
    else:
        size = im2.size(-2)
        phase = torch.atan2(im2[...,1], im2[...,0])  # (1, P_c, N, N)
        #shifted_phase = shift2(phase, loc, devid)  # (1, P_c, m, N, N)

        z = torch.arange(-size,size).repeat(tuple(loc.size()[:3])+(1,)).type(torch.cuda.FloatTensor).to(devid)  # (1, P_c, m, N)
        z1 = z.unsqueeze(-1).repeat(1, 1, 1, 1, 2*size)  # (1, P_c, m, N, N)
        z2 = z.unsqueeze(-2).repeat(1, 1, 1, 2*size, 1)  # (1, P_c, m, N, N)
        z = z1**2 + z2**2
        del z1; del z2
        z = unshift2(z, -torch.cuda.FloatTensor([size,size]).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1,loc.size(1),loc.size(2),1),
                     devid).to(devid)
        #z[:,:,:,0,0] = 1
        z = z.unsqueeze(-1)

        c = torch.cuda.FloatTensor([size/2,size/2]).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1,loc.size(1),loc.size(2),1).to(devid)
        shifted_phase = shift2(phase, loc+c, devid)
        cplx_phase = torch.stack((torch.cos(shifted_phase), torch.sin(shifted_phase)), dim=-1)

        cplx_phase_p1 = torch.cat((cplx_phase, cplx_phase.unsqueeze(-2).flip(-2,-3).squeeze().unsqueeze(0)),-2)
        cplx_phase_p = torch.cat((cplx_phase_p1, cplx_phase_p1.unsqueeze(-3).flip(-3,-4).squeeze().unsqueeze(0)),-3)
        del cplx_phase; del cplx_phase_p1
        cpi = complex_mul(torch.ifft(z*torch.fft(cplx_phase_p,2),2),conjugate(cplx_phase_p))[...,1]
        del cplx_phase_p
        z_ = z.clone()
        del z
    #    z_[:,:,:,size,size,:] = z_[:,:,:,size,size,:]+1
        lin_sp_c_dx = torch.ifft(torch.fft(torch.stack((cpi, 0*cpi.clone()),dim=-1),2)/z_, 2)[...,:size,:size,0]
        del cpi;
        lin_sp_c_dx_0 = lin_sp_c_dx - lin_sp_c_dx.clone()[:,:,:,size//2,size//2].unsqueeze(-1).unsqueeze(-1)
        lin_sp_c_dx_ = unshift2(lin_sp_c_dx_0, c, devid)
        del lin_sp_c_dx; del lin_sp_c_dx_0
        t_s_phase = torch.transpose(lin_sp_c_dx_, -1, -2)  # (1, P_c, m, N, N)
        del(shifted_phase)
        t_s_phase = torch.flip(t_s_phase.unsqueeze(-2), [-2,-3]).squeeze().unsqueeze(0)  # (1, P_c, m, N, N)
        orth_ph = unshift2(t_s_phase, loc, devid)  # (1, P_c, m, N, N)
        phase_ = unshift2(lin_sp_c_dx_, loc, devid)
        del(t_s_phase); del lin_sp_c_dx_
    return phase_, orth_ph

def orth_phase22(im2, loc, devid):
    """
    Given a batch of images and a tensor of local maxima, this function returns
    a tuple consisting of the phase and the orthogonal phase centered at the local
    minima.
    """
    #im2 (1, P_c, N, N, 2)
    size = im2.size(-2)
    phase = torch.atan2(im2[...,1], im2[...,0])  # (1, P_c, N, N)
    #  unwrapping
    z = torch.arange(-size//2,size//2).unsqueeze(0).unsqueeze(0)
    z = z.repeat(tuple(loc.size()[:2])+(1,)).type(torch.cuda.FloatTensor)  # (1, P_c, N)
    z1 = z.unsqueeze(-1).repeat(1, 1, 1, size)  # (1, P_c, N, N)
    z2 = z.unsqueeze(-2).repeat(1, 1, size, 1)  # (1, P_c, N, N)
    z = z1**2 + z2**2
    del z1; del z2
    z = shift2(z, -torch.cuda.FloatTensor([size//2,size//2]).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1,loc.size(1),1,1), devid)
    z = z.squeeze().unsqueeze(0).unsqueeze(-1)
    cplx_phase = torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1)
    cpi = complex_mul(torch.ifft(z*torch.fft(cplx_phase,2),2),conjugate(cplx_phase))[...,1]
    lin_sp_c_dx = torch.ifft(torch.fft(torch.stack((cpi, 0*cpi.clone()),dim=-1),2)/z, 2)[...,0]
    shifted_phase = shift2(phase, loc, devid)
    lin_sp_c_dx = lin_sp_c_dx - lin_sp_c_dx[:,:,0,0].unsqueeze(-1).unsqueeze(-1)
    lin_sp_c_dx = lin_sp_c_dx.unsqueeze(2) + shifted_phase[:,:,:,0,0].unsqueeze(-1).unsqueeze(-1)

    t_s_phase = torch.transpose(lin_sp_c_dx, -1, -2)  # (1, P_c, m, N, N)
    del(shifted_phase)
    if loc.size(2) == 1:
        t_s_phase = torch.flip(t_s_phase.unsqueeze(-2), [-2,-3]).squeeze().unsqueeze(0).unsqueeze(0)  # (1, P_c, m, N, N)
    else:
        t_s_phase = torch.flip(t_s_phase.unsqueeze(-2), [-2,-3]).squeeze().unsqueeze(0)  # (1, P_c, m, N, N)
    orth_ph = unshift2(t_s_phase, loc, devid)  # (1, P_c, m, N, N)
    phase_ = unshift2(lin_sp_c_dx, loc, devid)
    del(t_s_phase)
    return phase_, orth_ph

def shifted_phase(im2, loc, theta, devid):
    """
    theta: tensor of size (P_c)
    """
    size = im2.size(-2)
    theta_ = theta.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    theta_ = theta_.repeat(1, 1, loc.size(2), size, size)  # (1, P_c, m, N, N)
    phase, orth_ph = orth_phase22(im2, loc, devid)  # (1, P_c, N, N), (1, P_c, m, N, N)
    return torch.cos(theta_)*phase - torch.sin(theta_)*orth_ph  # (1, P_c, m, N, N)  (for orth_phase2)
#    return torch.cos(theta_)*phase.unsqueeze(2).repeat(1,1,loc.size(2),1,1) - torch.sin(theta_)*orth_ph  # (1, P_c, m, N, N)



def periodic_distance(x, y, N):
    return torch.min(torch.abs(x-y), torch.abs(x-y+N)).min(torch.abs(x-y-N))


def dist_to_max(u, size, devid):
    z = torch.arange(size, dtype = torch.float, device = devid).repeat(tuple(u.size()[:3])+(1,))  # (1, P_c, m, N)
    z1 = periodic_distance(z, u[..., 0].unsqueeze(3).repeat(1, 1, 1, size), size).unsqueeze(-1).repeat(1, 1, 1, 1, size)  # (1, P_c, m, N, N)
    z2 = periodic_distance(z, u[..., 1].unsqueeze(3).repeat(1, 1, 1, size), size).unsqueeze(-2).repeat(1, 1, 1, size, 1)  # (1, P_c, m, N, N)
    z1 = z1**2
    z2 = z2**2
    z = torch.sqrt(z1 + z2)  # (1, P_c, m, N, N)
    z_min = torch.min(z, dim=2)[0].unsqueeze(2).repeat(1, 1, u.size(2), 1, 1)  # (1, P_c, m, N, N)
    vors = torch.eq(z, z_min).type(torch.cuda.FloatTensor).to(devid)
    del(z); del(z_min); del(z1); del(z2)
    return vors

def new_phase(im1, im2, theta, ks, nb_centers, k1, k2, devid):
    # im1 and im2 (1, P_c, N, N, 2)
    k1 = torch.tensor(k1)
    k2 = torch.tensor(k2)
#    theta_ = theta.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, P_c, 1, 1)
    k = (1-k1.eq(0).type(torch.cuda.FloatTensor).to(devid))*(1-k2.eq(0).type(torch.cuda.FloatTensor).to(devid))  # (1, P_c, 1, 1)
#    k = k*torch.abs(theta_).eq(np.pi/2).type(torch.cuda.FloatTensor)  # (1, P_c, 1, 1)
#    k = 0*k
    k = k.repeat(1, 1, im1.size(-2), im1.size(-2))  # (1, P_c, N, N)
    im = im1.norm(p=2, dim=-1)*im2.norm(p=2, dim=-1)  # (1, P_c, N, N)
    #loc = local_max_indices(im, nb_centers)[1]  # (1, P_c, m, 2)
    loc = local_max_indices(im, ks, nb_centers, devid)[1]
    del(im)
    shifted_ph = shifted_phase(im2, loc, theta, devid)  # (1, P_c, m, N, N)
    vor = dist_to_max(loc, im1.size(-2), devid)  # (1, P_c, m, N, N)
    n_ph = (shifted_ph*vor).sum(dim=2) * k + torch.atan2(im2[...,1], im2[...,0]) * (1-k)
    return n_ph


def new_phase2(im1, im2, theta, loc, k1, k2, devid):
    # im1 and im2 (1, P_c, N, N, 2)
    k1 = torch.tensor(k1)
    k2 = torch.tensor(k2)
    k = (1-k1.eq(0).type(torch.cuda.FloatTensor).to(devid))*(1-k2.eq(0).type(torch.cuda.FloatTensor).to(devid))  # (1, P_c, 1, 1)
#    k = k*torch.abs(theta_).eq(np.pi/2).type(torch.cuda.FloatTensor)  # (1, P_c, 1, 1)
    k = k.repeat(1, 1, im1.size(-2), im1.size(-2))  # (1, P_c, N, N)
    shifted_ph = shifted_phase(im2, loc, theta, devid)  # (1, P_c, m, N, N)
    vor = dist_to_max(loc, im1.size(-2), devid)  # (1, P_c, m, N, N)
    n_ph = (shifted_ph*vor).sum(dim=2) * k + torch.atan2(im2[...,1], im2[...,0]) * (1-k)
    return n_ph


def phase_rot(im1, im2, theta, ks, nb_centers, k1, k2, devid):
    z = im2.norm(p=2, dim=-1)
    ph_rot = new_phase(im1, im2, theta, ks, nb_centers, k1, k2, devid)
    return torch.stack((z*torch.cos(ph_rot), z*torch.sin(ph_rot)), dim=-1)



class PhaseRot(object):
    def __init__(self): #, theta, ks, nb_centers, k1, k2, devid):
        #self.theta = theta
        #self.ks = ks
        #self.nb_centers = nb_centers
        #self.k1 = k1; self.k2 = k2
        #self.devid = devid
        self.loc = None


    def __call__(self, im1, im2, theta, ks, nb_centers, k1, k2, devid):
        if self.loc is None:
            z = im2.norm(p=2, dim=-1)
            im = im1.norm(p=2, dim=-1)*im2.norm(p=2, dim=-1)
            self.loc = local_max_indices(im, ks, nb_centers, devid)[1]
            ph_rot = new_phase(im1, im2, theta, ks, nb_centers, k1, k2, devid)
        else:
            z = im2.norm(p=2, dim=-1)
            ph_rot = new_phase2(im1, im2, theta, self.loc, k1, k2, devid)

        return torch.stack((z*torch.cos(ph_rot), z*torch.sin(ph_rot)), dim=-1)



def shift_mat():
    shift_m = torch.zeros(7,128)
    for j in range(7):
        v = torch.zeros(min(16*(2**j),128))
        for i in range((2**j)):
            v[i] = 1
        v = v.repeat(max(8//(2**j),1))
        shift_m[j,:] = v
    return shift_m

M = shift_mat()

def stack_permute(v,s1,s2):
    if s1<127:
        v_perm = torch.cat((v[:,s1:], v[:,:s1]), dim=-1)
        stack = torch.cat((v.repeat(s2,1), v_perm.repeat(s2,1)), dim=0)
    else:
        stack = v
    return stack

def matrix_rot(j):
    R = M[j,:].unsqueeze(0).repeat(128,1)
    for i in range(1,128//(2**j)):
        for k in range(2**j):
            R[i*(2**j)+k,:] = torch.cat((R[i*(2**j)+k,i*(2**j):], R[i*(2**j)+k,:i*(2**j)]))
    return R

