from .ST import *
from scipy.interpolate import interp1d
import time

def synthesis(
    estimator_name, target, image_init=None, image_ref=None, image_b=None,
    J=None, L=4, M=None, N=None,
    mode='image', optim_algorithm='LBFGS', steps=100, learning_rate=0.5,
    device='gpu', wavelets='morlet', savedir=None, seed=None,
    bispectrum_bins=None, bispectrum_bin_type='log',
    if_large_batch=False,
    C11_criteria=None,
    normalization='P00',
    precision='single',
    print_each_step=False,
):
    '''
the estimator_name can be 's_mean', 's_mean_iso', 's_cov', 's_cov_iso', 'alpha_cov', or 'bispectrum' the C11_criteria is the condition on j1 and j2 to compute coefficients, in addition to the condition that j2 >= j1. Use * or + to connect more than one condition.
    '''
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
                    st_calc = Scattering2d(
                        M, N, J, L, wavelets=wavelets, device=device, ref_a=target, ref_b=image_b
                    )
            else:
                st_calc = Scattering2d(M, N, J, L, wavelets=wavelets, device=device, ref=target)
        if mode=='estimator':
            if image_ref is None:
                st_calc = Scattering2d(M, N, J, L, wavelets=wavelets, device=device,)
                st_calc.add_synthesis_P00P11(target, 'iso' in estimator_name, C11_criteria)
            else:
                st_calc = Scattering2d(M, N, J, L, wavelets=wavelets, device=device, ref=image_ref)
        if estimator_name=='s_mean_iso':
            def func(image):
                return st_calc.scattering_coef(image, flatten=True)['for_synthesis_iso']
        if estimator_name=='s_mean':
            def func(image):
                return st_calc.scattering_coef(image, flatten=True)['for_synthesis']
        if estimator_name=='s_cov_iso':
            def func(image):
                return st_calc.scattering_cov(
                    image, if_synthesis=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria, 
                    normalization=normalization
                )['for_synthesis_iso']
        if estimator_name=='s_cov':
            def func(image):
                return st_calc.scattering_cov(
                    image, if_synthesis=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria,
                    normalization=normalization
                )['for_synthesis']
        if estimator_name=='s_cov_2fields_iso':
            def func(image):
                return st_calc.scattering_cov_2fields(
                    image, image_b, if_synthesis=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria,
                    normalization=normalization
                )['for_synthesis_iso']
        if estimator_name=='s_cov_2fields':
            def func(image):
                return st_calc.scattering_cov_2fields(
                    image, image_b, if_synthesis=True, if_large_batch=if_large_batch, C11_criteria=C11_criteria,
                    normalization=normalization
                )['for_synthesis']
    if 'alpha_cov' in estimator_name:
        aw_calc = PhaseHarmonics2d(M, N, J, L, wavelets=wavelets, device=device)
        def func(image):
            return aw_calc.forward(image)
    if 'bi' in estimator_name:
        if bispectrum_bins is None:
            bispectrum_bins = J-1
        if bispectrum_bin_type=='linear':
            k_range = np.linspace(1, M/2*1.4, bispectrum_bins) # linear binning
        if bispectrum_bin_type=='log':
            k_range = np.logspace(0,np.log10(M/2*1.4), bispectrum_bins) # log binning
        bi_calc = Bispectrum_Calculator(k_range, M, N, device=device)
        def func(image):
            return bi_calc.forward(image)
    
    # define loss function
    def quadratic_loss(target, model):
        return ((target - model)**2).mean()*1e8
    
    # synthesis
    image_syn = synthesis_general(
        target, image_init, func, quadratic_loss, 
        mode=mode, optim_algorithm=optim_algorithm, steps=steps, learning_rate=learning_rate,
        precision=precision, print_each_step=print_each_step
    )
    
    return image_syn

def synthesis_general(
    target, image_init, estimator_function, loss_function, 
    mode='image', optim_algorithm='LBFGS', steps=100, learning_rate=0.5,
    device='gpu', savedir=None, precision='single', print_each_step=False
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
        def __init__(self, image_init):
            super(OptimizableImage, self).__init__()
            self.param = torch.nn.Parameter( image_init.reshape(1,-1) )
    image_model = OptimizableImage(image_init)
        
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
            history_size=steps//2, line_search_fn=None
        )
    
    def closure():
        optimizer.zero_grad()
        loss = 0
        estimator_model = estimator_function(image_model.param.reshape(N_image,M,N))
        loss = loss_function(estimator_model, estimator_target)
        if print_each_step:
            if optim_algorithm=='LBFGS' or (optim_algorithm!='LBFGS' and (i%100== 0 or i%100==-1)):
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
        np.max((estimator_function(image_model.param.reshape(N_image,M,N)) - estimator_target).abs().detach().cpu().numpy()),
        ', mean residual: ', 
        np.mean((estimator_function(image_model.param.reshape(N_image,M,N)) - estimator_target).abs().detach().cpu().numpy()),
    )
    if optim_algorithm =='LBFGS':
        i=0
        optimizer.step(closure)
    else:
        for i in range(steps):
            print('step: ', i)
            optimizer.step(closure)
    print(
        'max residual: ', 
        np.max((estimator_function(image_model.param.reshape(N_image,M,N)) - estimator_target).abs().detach().cpu().numpy()),
        ', mean residual: ', 
        np.mean((estimator_function(image_model.param.reshape(N_image,M,N)) - estimator_target).abs().detach().cpu().numpy()),
    )
    t_end = time.time()
    print('time used: ', t_end - t_start, 's')

    return image_model.param.reshape(N_image,M,N).cpu().detach().numpy()


def standardize(image):
    return (image - image.mean((-2,-1)))/image.std((-2,-1))


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

def downsample(data):
    return (data[...,::2,::2] + data[...,1::2,::2] + data[...,::2,1::2] + data[...,1::2,1::2])/4




class model_image(torch.nn.Module):
    def __init__(
        self, N_realization, M, N, image, 
        mean_shift_init=0, power_factor_init=1, image_init=None, device='gpu'
    ):
        super(model_image, self).__init__()

        # initialize with GRF of same PS as target image
        image_to_train_numpy = np.zeros((N_realization, M, N))
        for n_realization in range(N_realization):
            image_to_train_numpy[n_realization] = get_random_data(
                    image[n_realization,:M,:N] - image[n_realization,:M,:N].mean(), 
                    M, N, "image"
                )
        image_to_train = (
            torch.from_numpy(image_to_train_numpy).reshape(
                N_realization, -1
            ) * power_factor_init
        ).type(torch.FloatTensor) + image.mean() + mean_shift_init
        if image_init is not None:
            image_to_train = torch.from_numpy(
            image_init.reshape(1,-1)
        ).type(torch.FloatTensor)
        if device=='gpu':
            image_to_train = image_to_train.cuda()
        self.param = torch.nn.Parameter( image_to_train )

def image_synthesis_ST(
    image, J, L, 
    N_realization=1,
    learnable_param_list = [(100, 1e-3)],
    other_function = None,
    savedir = '',
    device='gpu',
    coef = 'SC',
    random_seed = 987,
    wavelets='morlet',
    flip = True,
    optim_mode='LBFGS',
    mean_shift_init=0,
    power_factor_init = 1.0,
    image_init = None,
    low_bound = -0.010,
    high_bound = 1,
    bi_bins = 10,
    SC_use_ref = True,
):
    # define parameters
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    N_image = image.shape[0]
    M = image.shape[1]
    N = image.shape[2]
    if flip:
        flip_factor = 2
    else:
        flip_factor = 1
    
    # convert image into torch tensor and possibly add reflection padding
    image_torch = torch.from_numpy(image).type(torch.FloatTensor)
    if flip:
        image_flip_torch = torch.cat((
            torch.cat((
                image_torch, torch.flip(image_torch,[1])),1),
            torch.cat((
                torch.flip(image_torch,[2]), torch.flip(image_torch,[1,2])),1)), 2
        )
    else:
        image_flip_torch = image_torch
    if device=='gpu':
        image_torch = image_torch.cuda()
        image_flip_torch = image_flip_torch.cuda()
        
    # define calculators
    scattering_calculator = ST_2D(
        M*flip_factor, N*flip_factor, J, L, device=device, wavelets=wavelets,
        ref=image_flip_torch
    )
    bispectrum_calculator = Bispectrum_Calculator(
        # k_range = np.linspace(1, M/2*1.4, bi_bins), # linear binning
        k_range = np.logspace(0,np.log10(M/2*1.4), bi_bins), # log binning
        M=M*flip_factor, N=N*flip_factor, 
        device=device
    )

    # calculate statistics for target image
    target_ST_dict = scattering_calculator.scattering_coef(
        image_flip_torch, flatten=True
    )
    target_ST = target_ST_dict['all_iso']
    
    target_SC_dict = scattering_calculator.scattering_cov(
        image_flip_torch, flatten=True
    )
    target_Cov_all= target_SC_dict['all_iso']
    target_bi = bispectrum_calculator.forward(image_flip_torch)
    target_mean = image_flip_torch.mean((-2,-1))
    target_std  = image_flip_torch.std((-2,-1))
    print('# of ST: ', target_ST.shape[-1])
    print('# of SC: ', target_Cov_all.shape[-1])
    print('# of bi: ', target_bi.shape[-1])

    # cumulative distribution
    target_cumsum0 = image_torch.reshape(N_realization,-1).sort(-1).values
    target_cumsum = torch.from_numpy(np.empty((J, N_realization, M*N))).cuda()
    for j in range(J):
        for n_realization in range(N_realization):
            target_cumsum[j, n_realization] = smooth(
                image_torch[n_realization], j
            ).flatten().sort().values
    
    if other_function is not None:
        target_other_function = other_function(image_flip_torch)

    # define optimizable model
    model_fit = model_image(
        N_realization, M, N, image, 
        mean_shift_init,
        power_factor_init, 
        image_init, 
        device
    )

    # define statistics and optimize
    for learnable_group in range(len(learnable_param_list)):
        num_step = learnable_param_list[learnable_group][0]
        learning_rate = learnable_param_list[learnable_group][1]
        
        if optim_mode=='Adam':
            optimizer = torch.optim.Adam(model_fit.parameters(), lr=learning_rate)
        elif optim_mode=='NAdam':
            optimizer = torch.optim.NAdam(model_fit.parameters(), lr=learning_rate)
        elif optim_mode=='SGD':
            optimizer = torch.optim.SGD(model_fit.parameters(), lr=learning_rate)
        elif optim_mode=='Adamax':
            optimizer = torch.optim.Adamax(model_fit.parameters(), lr=learning_rate)
        elif optim_mode=='LBFGS':
            optimizer = torch.optim.LBFGS(
                model_fit.parameters(), lr=learning_rate, 
                max_iter=100, max_eval=None, 
                tolerance_grad=1e-09, tolerance_change=1e-09, 
                history_size=20, line_search_fn=None
            )
        
        def closure():
            optimizer.zero_grad()
            # Flipping continuation 
            if flip:
                image_param = model_fit.param.reshape(N_realization,M,N)
                image_param_flip = torch.cat((
                    torch.cat((
                        image_param, torch.flip(image_param,[1])),1),
                    torch.cat((
                        torch.flip(image_param,[2]), torch.flip(image_param,[1,2])),1)), 2
                )
            else:
                image_param = model_fit.param.reshape(N_realization,M,N)
                image_param_flip = image_param

            loss = 0
            loss_mean = (
                (target_mean/target_std - image_param_flip.mean((-2,-1))/target_std)**2
            ).sum()
            loss += loss_mean
            if 'hist' in coef:
                loss_hist = (
                    (
                        image_param.reshape(N_realization,-1).sort(dim=-1).values - 
                        target_cumsum0
                    )**2
                ).sum() / image_torch.var()
                loss += loss_hist
                if '7' in coef:
                    for j in range(J):
                        loss_hist7 += (
                            (
                                smooth(image_param,j).reshape(N_realization,-1).sort(dim=-1).values - 
                                target_cumsum[j]
                            )**2
                        ).sum() / smooth(image_torch,j).var()
                lost_hist *= 1 / M / N * 1e3
                loss += loss_hist
            if 'lbound' in coef:
                loss_lbound = torch.exp(
                    (5 + low_bound - image_param.reshape(N_realization,-1))/0.03
                ).mean()
                loss += loss_lbound
            if 'bi' in coef:
                bi = bispectrum_calculator.forward(
                    image_param_flip
                )
                loss_bi = ((target_bi - bi)**2).sum() / N_image
                loss += loss_bi
                # add power spectrum
                model_P00 = scattering_calculator.scattering_coef(
                    image_param_flip, flatten=True
                )['P00_iso'].log()
                loss_P00 = (
                    (target_ST_dict['P00_iso'].log() - model_P00)**2
                ).sum() / N_image
                loss += loss_P00
            if 'ST' in coef:
                model_ST = scattering_calculator.scattering_coef(
                    image_param_flip, flatten=True
                )['all_iso']
                loss_ST = ((target_ST - model_ST)**2).sum() / N_image
                loss += loss_ST
            if 'SC' in coef:
                model_SC_dict = scattering_calculator.scattering_cov(
                    image_param_flip, use_ref=SC_use_ref, flatten=True
                )
                # scattering cov altogether
                model_Cov_all = model_SC_dict['all_iso']
                loss_Cov_all = ((target_Cov_all - model_Cov_all)**2).sum() / N_image
                loss += loss_Cov_all
                # err_P00, err_S1, err_C01, err_P11, err_C11 = loss_scattering_cov(
                #     target_SC_dict, model_SC_dict
                # )
                # loss += err_C11 #err_P00 + err_P11 err_C01 + 
            if other_function is not None:
                loss_other = ((target_other - other_function(image_param_flip))**2).sum() / N_image
                loss += loss_other
                
            if i%100== 0 or i%100==-1 or optim_mode=='LBFGS':
                print(i)
                print('loss: ',loss)
                print('err_mean: ',loss_mean**0.5)
                if 'PS' in coef: print('loss_PS: ',loss_PS)
                if 'hist' in coef: print('loss_hist: ',loss_hist)
                if 'lbound' in coef: print('loss_lbound: ',loss_lbound)
                if 'hbound' in coef: print('loss_hbound: ',loss_hbound)
                if 'bi' in coef: print('err_bi: ',(loss_bi/target_bi.shape[-1])**0.5)
                if 'ST' in coef: 
                    print('err_ST: ',((target_ST - model_ST)**2).mean()**0.5)
                    print('loss_ST: ',loss_ST)
                if 'SC' in coef: 
                    err_P00, err_S1, err_C01, err_P11, err_C11 = loss_scattering_cov(
                        target_SC_dict, model_SC_dict
                    )
                    print('err_P00: ',err_P00)
                    print('err_S1: ',err_S1)
                    print('err_Corr01: ',err_C01)
                    print('err_P11: ',err_P11)
                    print('err_Corr11: ',err_C11)
                    # print('loss_cov_all: ', loss_Cov_all)
            loss.backward()
            return loss

        # optimize
        for i in range(int(num_step)):
            optimizer.step(closure)

    return model_fit.param.reshape(N_realization,M,N).cpu().detach().numpy()

def loss_scattering_cov(target_SC_dict, model_SC_dict, ):
    '''
        calculate respectively the loss from different subgroups of 
        scattering covariance. It is used in the function "image_synthesis".
    '''
    target_P00 = target_SC_dict['P00_iso'].log()
    target_S1  = target_SC_dict['S1_iso'].log()
    target_C01 = target_SC_dict['C01_iso']
    target_P11 = target_SC_dict['P11_iso'].log()
    target_C11 = target_SC_dict['Corr11_iso']
    
    model_P00 = model_SC_dict['P00_iso'].log()
    model_S1  = model_SC_dict['S1_iso'].log()
    model_C01 = model_SC_dict['C01_iso']
    model_P11 = model_SC_dict['P11_iso'].log()
    model_C11 = model_SC_dict['Corr11_iso']
    
    err_P00 = ((target_P00 - model_P00)**2).mean()**0.5
    err_S1 =  ((target_S1  - model_S1 )**2).mean()**0.5
    err_C01 = ((target_C01 - model_C01).abs()**2).mean()**0.5
    err_P11 = ((target_P11 - model_P11)**2).mean()**0.5
    err_C11 = ((target_C11 - model_C11).abs()**2).mean()**0.5
    
    return err_P00, err_S1, err_C01, err_P11, err_C11

