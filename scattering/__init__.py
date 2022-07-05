import os
dirpath = os.path.dirname(__file__)

import numpy as np
from pathlib import Path
import time
import torch
import matplotlib.pyplot as plt

from scattering.utils import to_numpy
from scattering.ST import FiltersSet, Scattering2d, Bispectrum_Calculator, AlphaScattering2d_cov, \
    get_power_spectrum, reduced_ST
from scattering.angle_transforms import FourierAngle
from scattering.scale_transforms import FourierScale


# synthesis
def synthesis(
    estimator_name, target, image_init=None, image_ref=None, image_b=None,
    J=None, L=4, M=None, N=None, l_oversampling=1,
    mode='image', optim_algorithm='LBFGS', steps=300, learning_rate=0.2,
    device='gpu', wavelets='morlet', seed=None,
    if_large_batch=False,
    C11_criteria=None,
    normalization='P00',
    precision='single',
    print_each_step=False,
    s_cov_func=None,
    s_cov_func_params=[],
    Fourier=False,
    target_full=None,
    ps=False, ps_bins=None, ps_bin_type='log',
    bi=False, bispectrum_bins=None, bispectrum_bin_type='log',
    hist=False,
    hist_j=False,
):
    '''
the estimator_name can be 's_mean', 's_mean_iso', 's_cov', 's_cov_iso', 'alpha_cov', or 'bispectrum' the C11_criteria is the condition on j1 and j2 to compute coefficients, in addition to the condition that j2 >= j1. Use * or + to connect more than one condition.
    '''
    if not torch.cuda.is_available(): device='cpu'
    np.random.seed(seed)
    if C11_criteria is None:
        C11_criteria = 'j2>=j1'
    if mode=='image':
        _, M, N = target.shape
        print('input_size: ', target.shape)
    if image_init is None:
        if mode=='image':
            image_init = np.random.normal(
                target.mean((-2,-1))[:,None,None],
                target.std((-2,-1))[:,None,None],
                target.shape
            )
        else:
            if M is None:
                print('please assign image size M and N.')
            # if 's_' in estimator_name:
            #     image_init = get_random_data(target[:,], target, M=M, N=N, N_image=target.shape[0], mode='func', seed=seed)
            image_init = np.random.normal(0,1,(target.shape[0],M,N))
    elif type(image_init) is str:
        if image_init=='random phase':
            image_init = get_random_data(target, seed=seed) # gaussian field
        
    if J is None:
        J = int(np.log2(min(M,N))) - 1
    
    # define calculator and estimator function
    if 's' in estimator_name:
        if mode=='image':
            if '2fields' in estimator_name:
                if image_b is None:
                    print('should provide a valid image_b.')
                else:
                    st_calc = Scattering2d(M, N, J, L, l_oversampling=l_oversampling, wavelets=wavelets, device=device, ref_a=target, ref_b=image_b)
            else:
                st_calc = Scattering2d(M, N, J, L, l_oversampling=l_oversampling, wavelets=wavelets, device=device, ref=target, )
        if mode=='estimator':
            if image_ref is None:
                st_calc = Scattering2d(M, N, J, L, l_oversampling=l_oversampling, wavelets=wavelets, device=device, )
                if target_full is None:
                    temp = target
                else:
                    temp = target_full
                st_calc.add_synthesis_P00P11(temp, 'iso' in estimator_name, C11_criteria)
            else:
                st_calc = Scattering2d(M, N, J, L, l_oversampling=l_oversampling, wavelets=wavelets, device=device, ref=image_ref, )
        if estimator_name=='s_mean_iso':
            func_s = lambda x: st_calc.scattering_coef(x, flatten=True)['for_synthesis_iso']
        if estimator_name=='s_mean':
            func_s = lambda x: st_calc.scattering_coef(x, flatten=True)['for_synthesis']
        if 's_cov_func' in estimator_name:
            def func_s(image):
                s_cov_set = st_calc.scattering_cov(
                    image, use_ref=True, if_large_batch=if_large_batch, 
                    C11_criteria=C11_criteria, 
                    normalization=normalization
                )
                return s_cov_func(s_cov_set, s_cov_func_params)
        if estimator_name=='s_cov_iso_para_perp':
            def func_s(image):
                result = st_calc.scattering_cov(
                    image, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria, 
                    normalization=normalization
                )
                index_type, j1, l1, j2, l2, j3, l3 = result['index_for_synthesis_iso']
                select = (index_type<3) + ((l2==0) + (l2==L//2)) * ((l3==0) + (l3==L//2) + (l3==-1))
                return result['for_synthesis_iso'][:,select]
        if estimator_name=='s_cov_iso_iso':
            def func_s(image):
                result = st_calc.scattering_cov(
                    image, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria, 
                    normalization=normalization
                )
                coef = result['for_synthesis_iso']
                index_type, j1, l1, j2, l2, j3, l3 = result['index_for_synthesis_iso']
                return torch.cat((
                    (coef[:,index_type<3]),
                    (coef[:,index_type==3].reshape(-1,L).mean(-1).reshape(len(coef),-1)),
                    (coef[:,index_type==4].reshape(-1,L).mean(-1).reshape(len(coef),-1)),
                    (coef[:,index_type==5].reshape(-1,L,L).mean((-2,-1)).reshape(len(coef),-1)),
                    (coef[:,index_type==6].reshape(-1,L,L).mean((-2,-1)).reshape(len(coef),-1)),
                ), dim=-1)
        if estimator_name=='s_cov_iso':
            func_s = lambda x: st_calc.scattering_cov(x, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria, normalization=normalization)['for_synthesis_iso']
        if estimator_name=='s_cov':
            func_s = lambda x: st_calc.scattering_cov(x, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria, normalization=normalization)['for_synthesis']
        if estimator_name=='s_cov_2fields_iso':
            def func_s(image):
                result = st_calc.scattering_cov_2fields(
                    image, image_b, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria,
                    normalization=normalization
                )
                select =(result['index_for_synthesis_iso'][0]!=1) * (result['index_for_synthesis_iso'][0]!=3) *\
                        (result['index_for_synthesis_iso'][0]!=7) * (result['index_for_synthesis_iso'][0]!=11)*\
                        (result['index_for_synthesis_iso'][0]!=15)* (result['index_for_synthesis_iso'][0]!=19)
                return result['for_synthesis_iso'][:,select]
        if estimator_name=='s_cov_2fields':
            def func_s(image):
                result = st_calc.scattering_cov_2fields(
                    image, image_b, use_ref=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria,
                    normalization=normalization
                )
                select =(result['index_for_synthesis'][0]!=1) * (result['index_for_synthesis'][0]!=3) *\
                        (result['index_for_synthesis'][0]!=7) * (result['index_for_synthesis'][0]!=11) *\
                        (result['index_for_synthesis'][0]!=15)* (result['index_for_synthesis'][0]!=19)
                return result['for_synthesis'][:,select]
    if 'alpha_cov' in estimator_name:
        aw_calc = AlphaScattering2d_cov(M, N, J, L, wavelets=wavelets, device=device)
        func_s = lambda x: aw_calc.forward(x)
    # power spectrum
    if ps:
        if ps_bins is None:
            ps_bins = J-1
        def func_ps(image):
            ps, _ = get_power_spectrum(image, bins=ps_bins, bin_type=ps_bin_type)
            return torch.cat(((image.mean((-2,-1))/image.std((-2,-1)))[:,None], ps), axis=-1)
    # bispectrum
    if bi:
        if bispectrum_bins is None:
            bispectrum_bins = J-1
        bi_calc = Bispectrum_Calculator(M, N, bins=bispectrum_bins, bin_type=bispectrum_bin_type, device=device)
        def func_bi(image):
            bi = bi_calc.forward(image)
            ps, _ = get_power_spectrum(image, bins=bispectrum_bins, bin_type=bispectrum_bin_type)
            return torch.cat(((image.mean((-2,-1))/image.std((-2,-1)))[:,None], ps, bi), axis=-1)
    # histogram
    def func_hist(image):
        flat_image = image.reshape(len(image),-1)
        return flat_image.sort(dim=-1).values.reshape(len(image),-1,image.shape[-2]).mean(-1) / flat_image.std(-1)[:,None]
    def smooth(image, j):
        M, N = image.shape[-2:]
        X = torch.arange(M)[:,None]
        Y = torch.arange(N)[None,:]
        R2 = (X-M//2)**2 + (Y-N//2)**2
        weight_f = torch.fft.fftshift(torch.exp(-0.5 * R2 / (M//(2**j)//2)**2)).cuda()
        image_smoothed = torch.fft.ifftn(torch.fft.fftn(image, dim=(-2,-1)) * weight_f[None,:,:], dim=(-2,-1))
        return image_smoothed.real
    def func_hist_j(image, J):
        cumsum_list = []
        flat_image = image.reshape(len(image),-1)
        cumsum_list.append(
            flat_image.sort(dim=-1).values.reshape(len(image),-1,image.shape[-2]).mean(-1) / flat_image.std(-1)[:,None]
        )
        for j in range(J):
            flat_image = smooth(image, j).reshape(len(image),-1)
            cumsum_list.append(
                flat_image.sort(dim=-1).values.reshape(len(image),-1,image.shape[-2]).mean(-1) / flat_image.std(-1)[:,None]
            )
        return torch.cat((cumsum_list), dim=-1)
    
    def func(image):
        coef_list = []
        if estimator_name!='':
            coef_list.append(func_s(image))
        if ps:
            coef_list.append(func_ps(image))
        if bi:
            coef_list.append(func_bi(image))
        if hist:
            coef_list.append(func_hist(image))
        if hist_j:
            coef_list.append(func_hist_j(image, J))
        return torch.cat(coef_list, axis=-1)
    
    # define loss function
    def quadratic_loss(target, model):
        return ((target - model)**2).mean()*1e8
    
    # synthesis
    image_syn = synthesis_general(
        target, image_init, func, quadratic_loss, 
        mode=mode, 
        optim_algorithm=optim_algorithm, steps=steps, learning_rate=learning_rate,
        device=device, precision=precision, print_each_step=print_each_step,
        Fourier=Fourier,
    )
    return image_syn

# manipulate output of flattened scattering_cov
def synthesis_general(
    target, image_init, estimator_function, loss_function, 
    mode='image', optim_algorithm='LBFGS', steps=100, learning_rate=0.5,
    device='gpu', precision='single', print_each_step=False, Fourier=False
):
    # define parameters
    N_image = image_init.shape[0]
    M = image_init.shape[1]
    N = image_init.shape[2]
    
    # formating target and image_init (to tensor, to cuda)
    if type(target)==np.ndarray:
        target = torch.from_numpy(target)
    if type(image_init)==np.ndarray:
        image_init = torch.from_numpy(image_init)
    if precision=='double':
        target = target.type(torch.DoubleTensor)
        image_init = image_init.type(torch.DoubleTensor)
    else:
        target = target.type(torch.FloatTensor)
        image_init = image_init.type(torch.FloatTensor)
    if device=='gpu':
        target     = target.cuda()
        image_init = image_init.cuda()
    
    # calculate statistics for target images
    if mode=='image':
        estimator_target = estimator_function(target)
    if mode=='estimator':
        estimator_target = target
    print('# of estimators: ', estimator_target.shape[-1])
    
    # define optimizable image model
    class OptimizableImage(torch.nn.Module):
        def __init__(self, input_init, Fourier=False):
            # super(OptimizableImage, self).__init__()
            super().__init__()
            self.param = torch.nn.Parameter( input_init )
            
            if Fourier: 
                self.image = torch.fft.ifftn(
                    self.param[0] + 1j*self.param[1],
                    dim=(-2,-1)).real
            else: self.image = self.param
    
    if Fourier: 
        temp = torch.fft.fftn(image_init, dim=(-2,-1))
        input_init = torch.cat((temp.real[None,...], temp.imag[None,...]), dim=0)
    else: input_init = image_init
    image_model = OptimizableImage(input_init, Fourier)
        
    # define optimizer
    if optim_algorithm   =='Adam':
        optimizer = torch.optim.Adam(image_model.parameters(), lr=learning_rate)
    elif optim_algorithm =='NAdam':
        optimizer = torch.optim.NAdam(image_model.parameters(), lr=learning_rate)
    elif optim_algorithm =='SGD':
        optimizer = torch.optim.SGD(image_model.parameters(), lr=learning_rate)
    elif optim_algorithm =='Adamax':
        optimizer = torch.optim.Adamax(image_model.parameters(), lr=learning_rate)
    elif optim_algorithm =='LBFGS':
        optimizer = torch.optim.LBFGS(image_model.parameters(), lr=learning_rate, 
            max_iter=steps, max_eval=None, 
            tolerance_grad=1e-19, tolerance_change=1e-19, 
            history_size=min(steps//2, 150), line_search_fn=None
        )
    
    def closure():
        optimizer.zero_grad()
        loss = 0
        estimator_model = estimator_function(image_model.image)
        loss = loss_function(estimator_model, estimator_target)
        if print_each_step:
            if optim_algorithm=='LBFGS' or (optim_algorithm!='LBFGS' and (i%100==0 or i%100==-1)):
                print((estimator_model-estimator_target).abs().max())
                print(
                    'max residual: ', 
                    np.max((estimator_model - estimator_target).abs().detach().cpu().numpy()),
                    ', mean residual: ', 
                    np.mean((estimator_model - estimator_target).abs().detach().cpu().numpy()),
                )
        loss.backward()
        return loss
    
    # optimize
    t_start = time.time()
    print(
        'max residual: ', 
        np.max((estimator_function(image_model.image) - estimator_target).abs().detach().cpu().numpy()),
        ', mean residual: ', 
        np.mean((estimator_function(image_model.image) - estimator_target).abs().detach().cpu().numpy()),
    )
    if optim_algorithm =='LBFGS':
        i=0
        optimizer.step(closure)
    else:
        for i in range(steps):
            # print('step: ', i)
            optimizer.step(closure)
    print(
        'max residual: ', 
        np.max((estimator_function(image_model.image) - estimator_target).abs().detach().cpu().numpy()),
        ', mean residual: ', 
        np.mean((estimator_function(image_model.image) - estimator_target).abs().detach().cpu().numpy()),
    )
    t_end = time.time()
    print('time used: ', t_end - t_start, 's')

    return image_model.image.cpu().detach().numpy()

# image pre-processing
def binning2x2(image):
    return (image[...,::2,::2] + image[...,1::2,::2] + image[...,::2,1::2] + image[...,1::2,1::2])/4

def whiten(image):
    return (image - image.mean((-2,-1))[:,None,None]) / image.std((-2,-1))[:,None,None]

def remove_slope(images):
    '''
        Removing the overall trend of an image by subtracting the result of
        a 2D linear fitting. This operation can reduce the edge effect when
        the field has too strong low-frequency components.
    '''
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

# get derivatives for a vector field
def get_w_div(u):
    vxy = np.roll(u[...,0],-1,1) - np.roll(u[...,0],1,1)
    vxz = np.roll(u[...,0],-1,2) - np.roll(u[...,0],1,2)
    vyx = np.roll(u[...,1],-1,0) - np.roll(u[...,1],1,0)
    vyz = np.roll(u[...,1],-1,2) - np.roll(u[...,1],1,2)
    vzx = np.roll(u[...,2],-1,0) - np.roll(u[...,2],1,0)
    vzy = np.roll(u[...,2],-1,1) - np.roll(u[...,2],1,1)
    vxx = np.roll(u[...,0],-1,0) - np.roll(u[...,0],1,0)
    vyy = np.roll(u[...,1],-1,1) - np.roll(u[...,1],1,1)
    vzz = np.roll(u[...,2],-1,2) - np.roll(u[...,2],1,2)
    
    wx = vzy - vyz
    wy = vxz - vzx
    wz = vyx - vxy
    div = vxx + vyy + vzz
    return np.array([wx, wy, wz, div]).transpose((1,2,3,4,0))

# get random initialization
def get_random_data(target, M=None, N=None, N_image=None, mode='image', seed=None):
    '''
    get a gaussian random field with the same power spectrum as the image 'target' (in the 'image' mode),
    or with an assigned power spectrum function 'target' (in the 'func' mode).
    '''
    np.random.seed(seed)
    if mode == 'image':
        N_image = target.shape[0]
        M = target.shape[-2]
        N = target.shape[-1]
        random_phase       = np.random.rand(N_image, M//2-1,N-1)
        random_phase_left  = np.random.rand(N_image, M//2-1, 1)
        random_phase_top   = np.random.rand(N_image, 1, N//2-1)
        random_phase_middle= np.random.rand(N_image, 1, N//2-1)
        random_phase_corners=np.random.randint(0,2,(N_image, 3))/2
    if mode == 'func':
        random_phase       = np.random.normal(0,1,(N_image,M//2-1,N-1)) + np.random.normal(0,1,(N_image,M//2-1,N-1))*1j
        random_phase_left  = (np.random.normal(0,1,(N_image,M//2-1,1)) + np.random.normal(0,1,(N_image,M//2-1,1))*1j)
        random_phase_top   = (np.random.normal(0,1,(N_image,1,N//2-1)) + np.random.normal(0,1,(N_image,1,N//2-1))*1j)
        random_phase_middle= (np.random.normal(0,1,(N_image,1,N//2-1)) + np.random.normal(0,1,(N_image,1,N//2-1))*1j)
        random_phase_corners=np.random.normal(0,1,(N_image,3))
        
    gaussian_phase = np.concatenate((
        np.concatenate((
            random_phase_corners[:,1,None,None], random_phase_left, random_phase_corners[:,2,None,None], -random_phase_left[:,::-1,:]
        ),axis=-2),
        np.concatenate((
            np.concatenate((random_phase_top, random_phase_corners[:,0,None,None], -random_phase_top[:,:,::-1]),axis=-1), random_phase,
            np.concatenate((random_phase_middle, np.zeros(N_image)[:,None,None], -random_phase_middle[:,:,::-1]),axis=-1), -random_phase[:,::-1,::-1],
        ),axis=-2),
    ),axis=-1)

    if mode == 'image':
        gaussian_modulus = np.abs(np.fft.fftshift(np.fft.fft2(target)))
        gaussian_field = np.fft.ifft2(np.fft.fftshift(gaussian_modulus*np.exp(1j*2*np.pi*gaussian_phase)))
    if mode == 'func':
        X = np.arange(0,M)
        Y = np.arange(0,N)
        Xgrid, Ygrid = np.meshgrid(X,Y, indexing='ij')
        R = ((Xgrid-M/2)**2+(Ygrid-N/2)**2)**0.5
        gaussian_modulus = target(R)
        gaussian_modulus[M//2, N//2] = 0
        gaussian_field = np.fft.ifft2(np.fft.fftshift(gaussian_modulus[None,:,:]*gaussian_phase))
    data = np.fft.fftshift(np.real(gaussian_field))
    return data

# transforming scattering representation s_cov['for_synthesis_iso']
# angular fft of C01 and C11
def fft_coef(coef, index_type, L):
    C01_f = torch.fft.fft(
        coef[:,index_type==3].reshape(len(coef),-1,L) +
        coef[:,index_type==4].reshape(len(coef),-1,L) * 1j, norm='ortho')
    C11_f =torch.fft.fft2(
        coef[:,index_type==5].reshape(len(coef),-1,L,L) +
        coef[:,index_type==6].reshape(len(coef),-1,L,L) * 1j, norm='ortho')
    return C01_f, C11_f

def s_cov_WN(st_calc, N_realization=500, coef_name='for_synthesis_iso'):
    '''
    compute the mean and std of C01_f and C11_f for gaussian white noise.
    '''
    image_gaussian = np.random.normal(0,1,(50,st_calc.M,st_calc.N))
    s_cov = st_calc.scattering_cov(image_gaussian, if_large_batch=True)
    coef = s_cov[coef_name]
    index_type, j1, j2, j3, l1, l2, l3 = s_cov['index_'+coef_name]

    for i in range(N_realization//50-1):
        image_gaussian = np.random.normal(0,1,(50,st_calc.M,st_calc.N))
        s_cov  = st_calc.scattering_cov(image_gaussian, if_large_batch=True)
        coef = torch.cat((coef, s_cov[coef_name]), dim=0)
    C01_f, C11_f = fft_coef(coef, index_type, st_calc.L)
    return C01_f.mean(0), C01_f.std(0), C11_f.mean(0), C11_f.std(0)

# select s_cov['for_synthesis_iso'] with mask
def s_cov_iso_threshold(s_cov, param_list):
    '''
    this can be any function that eats the s_cov from scattering.s_cov()
    and some other parameters, and then outputs a flattened torch tensor.
    The output is flattened instead of with size of [N_image, -1] because
    the mask for each image can be different.
    '''
    L = param_list[0]
    coef = s_cov['for_synthesis_iso']
    index_type, j1, j2, j3, l1, l2, l3 = s_cov['index_for_synthesis_iso']
    
    # Fourier transform for l2-l1 and l3-l1
    C01_f, C11_f = fft_coef(coef, index_type, L)
    return torch.cat((
            coef[:,index_type<3].reshape(-1), # mean, P, S1
            (C01_f[param_list[1]].reshape(-1)).real,
            (C01_f[param_list[1]].reshape(-1)).imag,
            (C11_f[param_list[2]].reshape(-1)).real,
            (C11_f[param_list[2]].reshape(-1)).imag,
        ), dim=0)

def modify_angular(s_cov_set, factor, C01=False, C11=False, keep_para=False):
    '''
    a function to change the angular oscillation of C01 and/or C11 by a factor
    '''
    index_type, j1, j2, j3, l1, l2, l3 = s_cov_set['index_for_synthesis_iso']
    L = s_cov_set['P00'].shape[-1]
    s_cov = s_cov_set['for_synthesis_iso']*1.
    N_img = len(s_cov)
    if keep_para:
        if C01:
            s_cov[:,index_type==3] += (
                s_cov[:,index_type==3].reshape(N_img,-1,L) - 
                s_cov[:,index_type==3].reshape(N_img,-1,L)[:,:,0:1]
            ).reshape(N_img,-1) * factor
            s_cov[:,index_type==4] += (
                s_cov[:,index_type==4].reshape(N_img,-1,L) - 
                s_cov[:,index_type==4].reshape(N_img,-1,L)[:,:,0:1]
            ).reshape(N_img,-1) * factor
        if C11:
            s_cov[:,index_type==5] += (
                s_cov[:,index_type==5].reshape(N_img,-1,L,L) - 
                s_cov[:,index_type==5].reshape(N_img,-1,L,L)[:,:,0:1,0:1]
            ).reshape(N_img,-1) * factor
            s_cov[:,index_type==6] += (
                s_cov[:,index_type==6].reshape(N_img,-1,L,L) - 
                s_cov[:,index_type==6].reshape(N_img,-1,L,L)[:,:,0:1,0:1]
            ).reshape(N_img,-1) * factor
    else:
        if C01:
            s_cov[:,index_type==3] += (
                s_cov[:,index_type==3].reshape(N_img,-1,L) - 
                s_cov[:,index_type==3].reshape(N_img,-1,L).mean(-1)[:,:,None]
            ).reshape(N_img,-1) * factor
            s_cov[:,index_type==4] += (
                s_cov[:,index_type==4].reshape(N_img,-1,L) - 
                s_cov[:,index_type==4].reshape(N_img,-1,L).mean(-1)[:,:,None]
            ).reshape(N_img,-1) * factor
        if C11:
            s_cov[:,index_type==5] += (
                s_cov[:,index_type==5].reshape(N_img,-1,L,L) - 
                s_cov[:,index_type==5].reshape(N_img,-1,L,L).mean((-2,-1))[:,:,None,None]
            ).reshape(N_img,-1) * factor
            s_cov[:,index_type==6] += (
                s_cov[:,index_type==6].reshape(N_img,-1,L,L) - 
                s_cov[:,index_type==6].reshape(N_img,-1,L,L).mean((-2,-1))[:,:,None,None]
            ).reshape(N_img,-1) * factor
    return s_cov

# show three panel plots
def show(image_target, image_syn, hist_range=(-2, 2), hist_bins=50):
    for i in range(len(image_target)):
        plt.figure(figsize=(9,3), dpi=200)
        plt.subplot(131) 
        plt.imshow(image_target[i], vmin=hist_range[0], vmax=hist_range[1])
        plt.xticks([]); plt.yticks([]); plt.title('original field')
        plt.subplot(132)
        plt.imshow(image_syn[i], vmin=hist_range[0], vmax=hist_range[1])
        plt.xticks([]); plt.yticks([]); plt.title('modeled field')
        plt.subplot(133); 
        plt.hist(image_target[i].flatten(), hist_bins, hist_range, histtype='step', label='target')
        plt.hist(   image_syn[i].flatten(), hist_bins, hist_range, histtype='step', label='synthesized')
        plt.yscale('log'); plt.legend(loc='lower center'); plt.title('histogram')
        plt.show()

# old code for synthesis
# class model_image(torch.nn.Module):
#     def __init__(
#         self, N_realization, M, N, image, 
#         mean_shift_init=0, power_factor_init=1, image_init=None, device='gpu'
#     ):
#         super(model_image, self).__init__()

#         # initialize with GRF of same PS as target image
#         image_to_train_numpy = np.zeros((N_realization, M, N))
#         for n_realization in range(N_realization):
#             image_to_train_numpy[n_realization] = get_random_data(
#                     image[n_realization,:M,:N] - image[n_realization,:M,:N].mean(), 
#                     M, N, "image"
#                 )
#         image_to_train = (
#             torch.from_numpy(image_to_train_numpy).reshape(
#                 N_realization, -1
#             ) * power_factor_init
#         ).type(torch.FloatTensor) + image.mean() + mean_shift_init
#         if image_init is not None:
#             image_to_train = torch.from_numpy(
#             image_init.reshape(1,-1)
#         ).type(torch.FloatTensor)
#         if device=='gpu':
#             image_to_train = image_to_train.cuda()
#         self.param = torch.nn.Parameter( image_to_train )

# def image_synthesis_ST(
#     image, J, L, 
#     N_realization=1,
#     learnable_param_list = [(100, 1e-3)],
#     other_function = None,
#     savedir = '',
#     device='gpu',
#     coef = 'SC',
#     random_seed = 987,
#     wavelets='morlet',
#     flip = True,
#     optim_mode='LBFGS',
#     mean_shift_init=0,
#     power_factor_init = 1.0,
#     image_init = None,
#     low_bound = -0.010,
#     high_bound = 1,
#     bi_bins = 10,
#     SC_use_ref = True,
# ):
#     # define parameters
#     torch.manual_seed(random_seed)
#     np.random.seed(random_seed)
#     N_image = image.shape[0]
#     M = image.shape[1]
#     N = image.shape[2]
#     if flip:
#         flip_factor = 2
#     else:
#         flip_factor = 1
    
#     # convert image into torch tensor and possibly add reflection padding
#     image_torch = torch.from_numpy(image).type(torch.FloatTensor)
#     if flip:
#         image_flip_torch = torch.cat((
#             torch.cat((
#                 image_torch, torch.flip(image_torch,[1])),1),
#             torch.cat((
#                 torch.flip(image_torch,[2]), torch.flip(image_torch,[1,2])),1)), 2
#         )
#     else:
#         image_flip_torch = image_torch
#     if device=='gpu':
#         image_torch = image_torch.cuda()
#         image_flip_torch = image_flip_torch.cuda()
        
#     # define calculators
#     scattering_calculator = ST_2D(
#         M*flip_factor, N*flip_factor, J, L, device=device, wavelets=wavelets,
#         ref=image_flip_torch
#     )
#     bispectrum_calculator = Bispectrum_Calculator(
#         # k_range = np.linspace(1, M/2*1.4, bi_bins), # linear binning
#         k_range = np.logspace(0,np.log10(M/2*1.4), bi_bins), # log binning
#         M=M*flip_factor, N=N*flip_factor, 
#         device=device
#     )

#     # calculate statistics for target image
#     target_ST_dict = scattering_calculator.scattering_coef(
#         image_flip_torch, flatten=True
#     )
#     target_ST = target_ST_dict['all_iso']
    
#     target_SC_dict = scattering_calculator.scattering_cov(
#         image_flip_torch, flatten=True
#     )
#     target_Cov_all= target_SC_dict['all_iso']
#     target_bi = bispectrum_calculator.forward(image_flip_torch)
#     target_mean = image_flip_torch.mean((-2,-1))
#     target_std  = image_flip_torch.std((-2,-1))
#     print('# of ST: ', target_ST.shape[-1])
#     print('# of SC: ', target_Cov_all.shape[-1])
#     print('# of bi: ', target_bi.shape[-1])

#     # cumulative distribution
#     target_cumsum0 = image_torch.reshape(N_realization,-1).sort(-1).values
#     target_cumsum = torch.from_numpy(np.empty((J, N_realization, M*N))).cuda()
#     for j in range(J):
#         for n_realization in range(N_realization):
#             target_cumsum[j, n_realization] = smooth(
#                 image_torch[n_realization], j
#             ).flatten().sort().values
    
#     if other_function is not None:
#         target_other_function = other_function(image_flip_torch)

#     # define optimizable model
#     model_fit = model_image(
#         N_realization, M, N, image, 
#         mean_shift_init,
#         power_factor_init, 
#         image_init, 
#         device
#     )

#     # define statistics and optimize
#     for learnable_group in range(len(learnable_param_list)):
#         num_step = learnable_param_list[learnable_group][0]
#         learning_rate = learnable_param_list[learnable_group][1]
        
#         if optim_mode=='Adam':
#             optimizer = torch.optim.Adam(model_fit.parameters(), lr=learning_rate)
#         elif optim_mode=='NAdam':
#             optimizer = torch.optim.NAdam(model_fit.parameters(), lr=learning_rate)
#         elif optim_mode=='SGD':
#             optimizer = torch.optim.SGD(model_fit.parameters(), lr=learning_rate)
#         elif optim_mode=='Adamax':
#             optimizer = torch.optim.Adamax(model_fit.parameters(), lr=learning_rate)
#         elif optim_mode=='LBFGS':
#             optimizer = torch.optim.LBFGS(
#                 model_fit.parameters(), lr=learning_rate, 
#                 max_iter=100, max_eval=None, 
#                 tolerance_grad=1e-09, tolerance_change=1e-09, 
#                 history_size=20, line_search_fn=None
#             )
        
#         def closure():
#             optimizer.zero_grad()
#             # Flipping continuation 
#             if flip:
#                 image_param = model_fit.param.reshape(N_realization,M,N)
#                 image_param_flip = torch.cat((
#                     torch.cat((
#                         image_param, torch.flip(image_param,[1])),1),
#                     torch.cat((
#                         torch.flip(image_param,[2]), torch.flip(image_param,[1,2])),1)), 2
#                 )
#             else:
#                 image_param = model_fit.param.reshape(N_realization,M,N)
#                 image_param_flip = image_param

#             loss = 0
#             loss_mean = (
#                 (target_mean/target_std - image_param_flip.mean((-2,-1))/target_std)**2
#             ).sum()
#             loss += loss_mean
#             if 'hist' in coef:
#                 loss_hist = (
#                     (
#                         image_param.reshape(N_realization,-1).sort(dim=-1).values - 
#                         target_cumsum0
#                     )**2
#                 ).sum() / image_torch.var()
#                 loss += loss_hist
#                 if '7' in coef:
#                     for j in range(J):
#                         loss_hist7 += (
#                             (
#                                 smooth(image_param,j).reshape(N_realization,-1).sort(dim=-1).values - 
#                                 target_cumsum[j]
#                             )**2
#                         ).sum() / smooth(image_torch,j).var()
#                 lost_hist *= 1 / M / N * 1e3
#                 loss += loss_hist
#             if 'lbound' in coef:
#                 loss_lbound = torch.exp(
#                     (5 + low_bound - image_param.reshape(N_realization,-1))/0.03
#                 ).mean()
#                 loss += loss_lbound
#             if 'bi' in coef:
#                 bi = bispectrum_calculator.forward(
#                     image_param_flip
#                 )
#                 loss_bi = ((target_bi - bi)**2).sum() / N_image
#                 loss += loss_bi
#                 # add power spectrum
#                 model_P00 = scattering_calculator.scattering_coef(
#                     image_param_flip, flatten=True
#                 )['P00_iso'].log()
#                 loss_P00 = (
#                     (target_ST_dict['P00_iso'].log() - model_P00)**2
#                 ).sum() / N_image
#                 loss += loss_P00
#             if 'ST' in coef:
#                 model_ST = scattering_calculator.scattering_coef(
#                     image_param_flip, flatten=True
#                 )['all_iso']
#                 loss_ST = ((target_ST - model_ST)**2).sum() / N_image
#                 loss += loss_ST
#             if 'SC' in coef:
#                 model_SC_dict = scattering_calculator.scattering_cov(
#                     image_param_flip, use_ref=SC_use_ref, flatten=True
#                 )
#                 # scattering cov altogether
#                 model_Cov_all = model_SC_dict['all_iso']
#                 loss_Cov_all = ((target_Cov_all - model_Cov_all)**2).sum() / N_image
#                 loss += loss_Cov_all
#                 # err_P00, err_S1, err_C01, err_P11, err_C11 = loss_scattering_cov(
#                 #     target_SC_dict, model_SC_dict
#                 # )
#                 # loss += err_C11 #err_P00 + err_P11 err_C01 + 
#             if other_function is not None:
#                 loss_other = ((target_other - other_function(image_param_flip))**2).sum() / N_image
#                 loss += loss_other
                
#             if i%100== 0 or i%100==-1 or optim_mode=='LBFGS':
#                 print(i)
#                 print('loss: ',loss)
#                 print('err_mean: ',loss_mean**0.5)
#                 if 'PS' in coef: print('loss_PS: ',loss_PS)
#                 if 'hist' in coef: print('loss_hist: ',loss_hist)
#                 if 'lbound' in coef: print('loss_lbound: ',loss_lbound)
#                 if 'hbound' in coef: print('loss_hbound: ',loss_hbound)
#                 if 'bi' in coef: print('err_bi: ',(loss_bi/target_bi.shape[-1])**0.5)
#                 if 'ST' in coef: 
#                     print('err_ST: ',((target_ST - model_ST)**2).mean()**0.5)
#                     print('loss_ST: ',loss_ST)
#                 if 'SC' in coef: 
#                     err_P00, err_S1, err_C01, err_P11, err_C11 = loss_scattering_cov(
#                         target_SC_dict, model_SC_dict
#                     )
#                     print('err_P00: ',err_P00)
#                     print('err_S1: ',err_S1)
#                     print('err_Corr01: ',err_C01)
#                     print('err_P11: ',err_P11)
#                     print('err_Corr11: ',err_C11)
#                     # print('loss_cov_all: ', loss_Cov_all)
#             loss.backward()
#             return loss

#         # optimize
#         for i in range(int(num_step)):
#             optimizer.step(closure)

#     return model_fit.param.reshape(N_realization,M,N).cpu().detach().numpy()

# def loss_scattering_cov(target_SC_dict, model_SC_dict, ):
#     '''
#         calculate respectively the loss from different subgroups of 
#         scattering covariance. It is used in the function "image_synthesis".
#     '''
#     target_P00 = target_SC_dict['P00_iso'].log()
#     target_S1  = target_SC_dict['S1_iso'].log()
#     target_C01 = target_SC_dict['C01_iso']
#     target_P11 = target_SC_dict['P11_iso'].log()
#     target_C11 = target_SC_dict['Corr11_iso']
    
#     model_P00 = model_SC_dict['P00_iso'].log()
#     model_S1  = model_SC_dict['S1_iso'].log()
#     model_C01 = model_SC_dict['C01_iso']
#     model_P11 = model_SC_dict['P11_iso'].log()
#     model_C11 = model_SC_dict['Corr11_iso']
    
#     err_P00 = ((target_P00 - model_P00)**2).mean()**0.5
#     err_S1 =  ((target_S1  - model_S1 )**2).mean()**0.5
#     err_C01 = ((target_C01 - model_C01).abs()**2).mean()**0.5
#     err_P11 = ((target_P11 - model_P11)**2).mean()**0.5
#     err_C11 = ((target_C11 - model_C11).abs()**2).mean()**0.5
    
#     return err_P00, err_S1, err_C01, err_P11, err_C11


def scale_annotation_a_b(idx_info):
    """
    Convert idx_info j1, j1p, j2, l1, l1p, l2
    into idx_info j1, a, b, l1, l1p, l2.

    :idx_info: K x 6 array
    """
    cov_type, j1, j1p, j2, l1, l1p, l2 = idx_info.T
    admissible_types = {
        0: 'mean',
        1: 'P00',
        2: 'S1',
        3: 'C01re',
        4: 'C01im',
        5: 'C11re',
        6: 'C11im'
    }
    cov_type = np.array([admissible_types[c_type] for c_type in cov_type])

    # create idx_info j1, j1p, a, b, l1, l1p, l2
    where_c01_c11 = np.isin(cov_type, ['C01re', 'C01im', 'C11re', 'C11im'])

    j1_new = j1.copy()
    j1p_new = j1p.copy()

    j1_new[where_c01_c11] = j1p[where_c01_c11]
    j1p_new[where_c01_c11] = j1[where_c01_c11]

    a = (j1_new - j1p_new) * (j1p_new >= 0) - (j1p_new == -1)
    b = (j1_new - j2) * (j2 >= 0) + (j2 == -1)

    idx_info_a_b = np.array([cov_type, j1_new, a, b, l1, l1p, l2], dtype=object).T

    # idx_info_a_b = np.stack([cov_type, j1_new, a, b, l1, l1p, l2]).T

    return idx_info_a_b


if __name__ == "__main__":

    fourier_angle = True
    fourier_scale = True

    angle_operator = FourierAngle()
    scale_operator = FourierScale()

    def moments(s_cov, params):
        idx_info = to_numpy(s_cov['index_for_synthesis_iso']).T
        idx_info = scale_annotation_a_b(idx_info)
        s_cov = s_cov['for_synthesis_iso']

        if fourier_angle:
            s_cov, idx_info = angle_operator(s_cov, idx_info)
        if fourier_scale:
            s_cov, idx_info = scale_operator(s_cov, idx_info)

        return s_cov

    im1_path = Path(dirpath) / 'example_fields.npy'
    im = np.load(str(im1_path))

    image_syn = synthesis('s_cov_func', im[:1, :, :], s_cov_func=moments, J=7, steps=100, seed=0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), squeeze=False)
    axes[0, 0].imshow(im[0, :, :], cmap='viridis')
    axes[0, 0].grid(None)
    axes[0, 1].imshow(image_syn[0, :, :], cmap='viridis')
    axes[0, 1].grid(None)
    plt.show()

    print(0)

