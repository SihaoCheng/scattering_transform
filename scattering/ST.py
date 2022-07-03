import numpy as np
import torch
import torch.fft

class FiltersSet(object):
    def __init__(self, M, N, J, L):
        self.M = M
        self.N = N
        self.J = J
        self.L = L
    
    # Morlet Wavelets
    def generate_morlet(self, if_save=False, save_dir=None, precision='single', l_oversampling=1):
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
                    slant=4.0 / self.L * l_oversampling,
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
            ) * (self.M * self.N)**0.5
        if precision=='single':
            phi = torch.from_numpy(
                self.gabor_2d_mycode(self.M, self.N, 0.8 * 2**(self.J-1), 0, 0).real.astype(np.float32)
            ) * (self.M * self.N)**0.5
        
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
    
    # Bump Steerable Wavelet
    def generate_bump_steerable(self, if_save=False, save_dir=None, precision='single'):
        if precision=='double':
            psi = torch.zeros((self.J, self.L, self.M, self.N), dtype=torch.float64)
        if precision=='single':
            psi = torch.zeros((self.J, self.L, self.M, self.N), dtype=torch.float32)
        for j in range(self.J):
            for l in range(self.L):
                wavelet_Fourier = self.bump_steerable_2d(
                    M=self.M,
                    N=self.N, 
                    k0= 0.375 * 2 * np.pi / 2**j,
                    theta0=(int(self.L-self.L/2-1)-l) * np.pi / self.L, 
                    L=self.L
                )
                wavelet_Fourier[0,0] = 0
                if precision=='double':
                    psi[j, l] = torch.from_numpy(wavelet_Fourier)
                if precision=='single':
                    psi[j, l] = torch.from_numpy(wavelet_Fourier.astype(np.float32))
        if precision=='double':
            phi = torch.from_numpy(
                self.gabor_2d_mycode(self.M, self.N, 2 * np.pi /(0.702*2**(-0.05)) * 2**(self.J-1), 0, 0).real
            ) * (self.M * self.N)**0.5
        if precision=='single':
            phi = torch.from_numpy(
                self.gabor_2d_mycode(self.M, self.N, 2 * np.pi /(0.702*2**(-0.05)) * 2**(self.J-1), 0, 0).real.astype(np.float32)
            ) * (self.M * self.N)**0.5
        
        filters_set = {'psi':psi, 'phi':phi}
        if if_save:
            np.save(
                save_dir + 'filters_set_mycode_M' + str(self.M) + 'N' + str(self.N)
                + 'J' + str(self.J) + 'L' + str(self.L) + '_' + precision + '.npy', 
                np.array([{'filters_set': filters_set}])
            )
        return filters_set

    def bump_steerable_2d(self, M, N, k0, theta0, L):
        """
            (from kymatio package) 
            Computes a 2D bump steerable filter.
            A bump steerable filter is a filter defined with
            compact support in Fourier space in the range
            k in [0, 2*k0]. It is a steerable filter, meaning
            that its profile in Fourier space can be expressed
            as a radial part multiplied by an angular part:
            psi_fft(k_vec) = c * radial_part(k) * angular_part(theta),
            where c is a normalization constant: c = 1/1.29 * 2^(L/2-1) * (L/2-1)! / sqrt((L/2)(L-2)!),
            the radial profile is:  exp[ (-(k - k0)^2) / (k0^2 - (k - k0)^2) ], for k within [0, 2*k0]
            the angular profile is: (cos(theta - theta0)) ^ (L/2 - 1), for theta within [theta0-pi/2, theta0+pi/2].
            Parameters
            ----------
            M, N : int
                spatial sizes
            k0 : float
                central frequency (in [0, 1])
            theta0 : float
                angle in [0, pi]
            Returns
            -------
            bump_steerable_fft : ndarray
                numpy array of size (M, N)
        """
        xx = np.empty((2,2, M, N))
        yy = np.empty((2,2, M, N))

        for ii, ex in enumerate([-1, 0]):
            for jj, ey in enumerate([-1, 0]):
                xx[ii,jj], yy[ii,jj] = np.mgrid[
                    ex * M : M + ex * M, 
                    ey * N : N + ey * N
                ]
        k = ((xx/M)**2 + (yy/N)**2)**0.5 * 2 * np.pi
        theta = np.arctan2(yy, xx)
        
        radial = np.exp(-(k - k0)**2 / (2*k*k0 - k**2)) #(k0**2 - (k-k0)**2)) #
        radial[k==0] = 0
        radial[k>= 2 * k0] = 0
        angular = np.cos(theta - theta0)**(L/2-1+2)
        angular[np.cos(theta - theta0)<0] = 0
        c = 1/1.29 * 2**(L/2-1) * np.math.factorial(L/2-1) / np.sqrt((L/2)* np.math.factorial(L-2))
        bump_steerable_fft = c * (radial * angular).sum((0,1))
        return bump_steerable_fft

    # Gaussian Steerable Wavelet
    def generate_gau_steerable(self, if_save=False, save_dir=None, precision='single'):
        if precision=='double':
            psi = torch.zeros((self.J, self.L, self.M, self.N), dtype=torch.float64)
        if precision=='single':
            psi = torch.zeros((self.J, self.L, self.M, self.N), dtype=torch.float32)
        for j in range(self.J):
            for l in range(self.L):
                wavelet_Fourier = self.gau_steerable_2d(
                    M=self.M,
                    N=self.N, 
                    k0= 0.375 * 2 * np.pi / 2**j,
                    theta0=(int(self.L-self.L/2-1)-l) * np.pi / self.L, 
                    L=self.L
                )
                wavelet_Fourier[0,0] = 0
                if precision=='double':
                    psi[j, l] = torch.from_numpy(wavelet_Fourier)
                if precision=='single':
                    psi[j, l] = torch.from_numpy(wavelet_Fourier.astype(np.float32))
        if precision=='double':
            phi = torch.from_numpy(
                self.gabor_2d_mycode(self.M, self.N, 2 * np.pi /(0.702*2**(-0.05)) * 2**(self.J-1), 0, 0).real
            ) * (self.M * self.N)**0.5
        if precision=='single':
            phi = torch.from_numpy(
                self.gabor_2d_mycode(self.M, self.N, 2 * np.pi /(0.702*2**(-0.05)) * 2**(self.J-1), 0, 0).real.astype(np.float32)
            ) * (self.M * self.N)**0.5
        
        filters_set = {'psi':psi, 'phi':phi}
        if if_save:
            np.save(
                save_dir + 'filters_set_mycode_M' + str(self.M) + 'N' + str(self.N)
                + 'J' + str(self.J) + 'L' + str(self.L) + '_' + precision + '.npy', 
                np.array([{'filters_set': filters_set}])
            )
        return filters_set
        
    def gau_steerable_2d(self, M, N, k0, theta0, L):
        xx = np.empty((2,2, M, N))
        yy = np.empty((2,2, M, N))

        for ii, ex in enumerate([-1, 0]):
            for jj, ey in enumerate([-1, 0]):
                xx[ii,jj], yy[ii,jj] = np.mgrid[
                    ex * M : M + ex * M, 
                    ey * N : N + ey * N
                ]
        k = ((xx/M)**2 + (yy/N)**2)**0.5 * 2 * np.pi
        theta = np.arctan2(yy, xx)
        
        # radial = np.exp(-(k - k0)**2 / (2*k*k0 - k**2)) #(k0**2 - (k-k0)**2)) #
        radial = (2*k/k0)**2 * np.exp(-k**2/(2 * (k0/1.4)**2))
        radial[k==0] = 0
        # radial2 = np.exp(-(k - k0)**2 / (2 * (k0/2)**2))
        # radial[k>= k0] = radial2[k>= k0]
        angular = np.cos(theta - theta0)**(L/2-1+2)
        angular[np.cos(theta - theta0)<0] = 0
        c = 1/1.29 * 2**(L/2-1) * np.math.factorial(L/2-1) / np.sqrt((L/2)* np.math.factorial(L-2))
        gau_steerable_fft = c * (radial * angular).sum((0,1))
        return gau_steerable_fft

    # tophat (Shannon) Wavelet
    def generate_shannon(self, if_save=False, save_dir=None):
        psi = torch.zeros((self.J, self.L, self.M, self.N), dtype=torch.bool)
        for j in range(self.J):
            for l in range(self.L):
                wavelet_Fourier = self.shannon_2d(
                    M=self.M,
                    N=self.N, 
                    kmin= 0.375 * 2 * np.pi / 2**j / 2**0.5,
                    kmax= 0.375 * 2 * np.pi / 2**j * 2**0.5,
                    theta0=(int(self.L-self.L/2-1)-l) * np.pi / self.L, 
                    L=self.L
                )
                wavelet_Fourier[0,0] = False
                psi[j, l] = torch.from_numpy(wavelet_Fourier)
        phi = torch.from_numpy(
            self.shannon_2d(
                M=self.M,
                N=self.N, 
                kmin = -1,
                kmax = 0.375 * 2 * np.pi / 2**self.J * 2**0.5,
                theta0 = 0,
                L = 0.5
            )
        )
        filters_set = {'psi':psi, 'phi':phi}
        if if_save:
            np.save(
                save_dir + 'filters_set_mycode_M' + str(self.M) + 'N' + str(self.N)
                + 'J' + str(self.J) + 'L' + str(self.L) + '_' + precision + '.npy', 
                np.array([{'filters_set': filters_set}])
            )
        return filters_set
    
    def shannon_2d(self, M, N, kmin, kmax, theta0, L):
        xx = np.empty((2,2, M, N))
        yy = np.empty((2,2, M, N))

        for ii, ex in enumerate([-1, 0]):
            for jj, ey in enumerate([-1, 0]):
                xx[ii,jj], yy[ii,jj] = np.mgrid[
                    ex * M : M + ex * M,
                    ey * N : N + ey * N
                ]
        k = ((xx/M)**2 + (yy/N)**2)**0.5 * 2 * np.pi
        theta = np.arctan2(yy, xx)
        
        radial = (k > kmin) * (k <= kmax)
        # radial[k==0] = False
        angular = (np.remainder(theta0-theta, 2*np.pi) <  np.pi/L/2) + \
                  (np.remainder(theta-theta0, 2*np.pi) <= np.pi/L/2)
        # angular[np.cos(theta - theta0)<0] = 0
        tophat_fft = (radial * angular).sum((0,1)) > 0
        return tophat_fft
        
    #  gau_harmonic
    def generate_gau_harmonic(self, if_save=False, save_dir=None, precision='single'):
        if precision=='double':
            psi = torch.zeros((self.J, self.L, self.M, self.N), dtype=torch.complex128)
        if precision=='single':
            psi = torch.zeros((self.J, self.L, self.M, self.N), dtype=torch.complex64)
        # psi
        for j in range(self.J):
            for l in range(self.L):
                wavelet_Fourier = self.gau_harmonic_2d(
                    M=self.M,
                    N=self.N, 
                    k0= 0.375 * 2 * np.pi / 2**j,
                    l=l,
                )
                wavelet_Fourier[0,0] = 0
                if precision=='double':
                    psi[j,l] = torch.from_numpy(wavelet_Fourier)
                if precision=='single':
                    psi[j,l] = torch.from_numpy(wavelet_Fourier.astype(np.complex64))
        # phi
        if precision=='double':
            phi = torch.from_numpy(
                self.gabor_2d_mycode(self.M, self.N, 2 * np.pi /(0.702*2**(-0.05)) * 2**(self.J-1), 0, 0).real
            ) * (self.M * self.N)**0.5
        if precision=='single':
            phi = torch.from_numpy(
                self.gabor_2d_mycode(self.M, self.N, 2 * np.pi /(0.702*2**(-0.05)) * 2**(self.J-1), 0, 0).real.astype(np.float32)
            ) * (self.M * self.N)**0.5
        
        filters_set = {'psi':psi, 'phi':phi}
        if if_save:
            np.save(
                save_dir + 'filters_set_mycode_M' + str(self.M) + 'N' + str(self.N)
                + 'J' + str(self.J) + 'L' + str(self.L) + '_' + precision + '.npy', 
                np.array([{'filters_set': filters_set}])
            )
        return filters_set
        
    def gau_harmonic_2d(self, M, N, k0, l):
        xx = np.empty((2,2, M, N))
        yy = np.empty((2,2, M, N))

        for ii, ex in enumerate([-1, 0]):
            for jj, ey in enumerate([-1, 0]):
                xx[ii,jj], yy[ii,jj] = np.mgrid[
                    ex * M : M + ex * M, 
                    ey * N : N + ey * N
                ]
        k = ((xx/M)**2 + (yy/N)**2)**0.5 * 2 * np.pi
        theta = np.arctan2(yy, xx)
        
        # profile = radial * angular
        wavelet_fft = (2*k/k0)**2 * np.exp(-k**2/(2 * (k0/1.4)**2)) * np.exp(1j * l * theta)
        wavelet_fft[k==0] = 0
        return wavelet_fft.sum((0,1))

class Scattering2d(object):
    def __init__(
        self, M, N, J, L=4, device='gpu', 
        wavelets='morlet', filters_set=None, weight=None, 
        precision='single', ref=None, ref_a=None, ref_b=None,
        l_oversampling=1,
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
            if wavelets=='morlet':
                filters_set = FiltersSet(
                    M=M, N=N, J=J, L=L,
                ).generate_morlet(precision=precision, l_oversampling=l_oversampling)
            if wavelets=='BS':
                filters_set = FiltersSet(
                    M=M, N=N, J=J, L=L,
                ).generate_bump_steerable(precision=precision)
            if wavelets=='gau':
                filters_set = FiltersSet(
                    M=M, N=N, J=J, L=L,
                ).generate_gau_steerable(precision=precision)
            if wavelets=='shannon':
                filters_set = FiltersSet(
                    M=M, N=N, J=J, L=L,
                ).generate_shannon()
            if wavelets=='gau_harmonic':
                filters_set = FiltersSet(
                    M=M, N=N, J=J, L=L,
                ).generate_gau_harmonic()
            self.M, self.N = M, N
        else:
            self.M, self.N = filters_set['psi'][0][0].shape
        self.J, self.L = J, L
        
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
                    self.cut_high_k_off(self.weight_f, dx, dy),
                    dim=(-2,-1)
                ).real
                if device=='gpu':
                    weight_downsample = weight_downsample.cuda()
                self.weight_downsample_list.append(
                    weight_downsample / weight_downsample.mean()
                )
        self.edge_masks = torch.empty((J, self.M, self.N))
        X, Y = torch.meshgrid(torch.arange(self.M), torch.arange(self.N), indexing='ij')
        for j in range(J):
            self.edge_masks[j] = (X>=2**j*2)*(X<=self.M-2**j*2)*\
                    (Y>=2**j*2)*(Y<=self.N-2**j*2)
        
        # device
        self.device = device
        if device=='gpu':
            self.filters_set = self.filters_set.cuda()
            self.phi = self.phi.cuda()
            if weight is not None:
                self.weight = self.weight.cuda()
            self.edge_masks = self.edge_masks.cuda()
        
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
    
    def add_synthesis_P00P11(self, s_cov, if_iso, C11_criteria='j2>=j1'):
        J = self.J
        L = self.L
        self.ref_scattering_cov = {}
        if if_iso:
            j1, j2, l2 = torch.meshgrid(torch.arange(J), torch.arange(J), torch.arange(L), indexing='ij')
            select_j12_iso = (j1 <= j2) * eval(C11_criteria)
            self.ref_scattering_cov['P00'] = torch.exp(s_cov[:,1:1+J].reshape((-1,J,1)))
            self.ref_scattering_cov['P11'] = torch.zeros(s_cov.shape[0], J,J,L,L)
            for i in range(select_j12_iso.sum()):
                self.ref_scattering_cov['P11'][:,j1[select_j12_iso][i],j2[select_j12_iso][i],:,l2[select_j12_iso][i]] = \
                    torch.exp(s_cov[:,1+2*J+i,None])
        else:
            j1, j2, l1, l2 = torch.meshgrid(torch.arange(J), torch.arange(J), torch.arange(L), torch.arange(L), indexing='ij')
            select_j12 = (j1 <= j2) * eval(C11_criteria)
            self.ref_scattering_cov['P00'] = torch.exp(s_cov[:,1:1+J*L].reshape((-1,J,L)))
            self.ref_scattering_cov['P11'] = torch.zeros(s_cov.shape[0], J,J,L,L)
            for i in range(select_j12.sum()):
                self.ref_scattering_cov['P11'][
                    :,j1[select_j12][i],j2[select_j12][i],l1[select_j12][i],l2[select_j12][i]
                ] = torch.exp(s_cov[:,1+2*J*L+i])
        if self.device=='gpu':
            self.ref_scattering_cov['P00'] = self.ref_scattering_cov['P00'].cuda()
            self.ref_scattering_cov['P11'] = self.ref_scattering_cov['P11'].cuda()
        
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
    
    def get_index(self, J, L, normalization='P00', C11_criteria='j1>-1', num_field=1):
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
                index_2_iso, index_2_iso, index_3_iso, index_3_iso, # C01re, C01im, C11re, C11im
            ), dim=-1)
            # label different coef
            index_type = torch.cat((
                index_mean[:1,:]*0, index_1[:1,:]*0+1, index_1[:1,:]*0+2,
                index_2[:1,:]*0+3,  index_2[:1,:]*0+4, index_3[:1,:]*0+5, index_3[:1,:]*0+6
            ), dim=-1)
            index_type_iso = torch.cat((
                index_mean[:1,:]*0,    index_1_iso[:1,:]*0+1, index_1_iso[:1,:]*0+2,
                index_2_iso[:1,:]*0+3, index_2_iso[:1,:]*0+4, index_3_iso[:1,:]*0+5, index_3_iso[:1,:]*0+6
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
                index_2_iso, index_2_iso, index_2_iso, index_2_iso, # C01im
                index_3_iso, index_3_iso, index_3_iso, index_3_iso, # C11re
                index_3_iso, index_3_iso, index_3_iso, index_3_iso, # C11im
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
                index_2_iso[:1,:]*0+10, index_2_iso[:1,:]*0+11,index_2_iso[:1,:]*0+12,index_2_iso[:1,:]*0+13,
                index_3_iso[:1,:]*0+14, index_3_iso[:1,:]*0+15,index_3_iso[:1,:]*0+16,index_3_iso[:1,:]*0+17,
                index_3_iso[:1,:]*0+18, index_3_iso[:1,:]*0+19,index_3_iso[:1,:]*0+20,index_3_iso[:1,:]*0+21,
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
                    data_f_small = self.cut_high_k_off(data_f, dx1, dy1)
                    wavelet_f = self.cut_high_k_off(filters_set[j1], dx1, dy1)
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
                            I1_f_small = self.cut_high_k_off(I1_f, dx2, dy2)
                            wavelet_f2 = self.cut_high_k_off(filters_set[j2], dx2, dy2)
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
                    data_f_small = self.cut_high_k_off(data_f, dx1, dy1)
                    wavelet_f = self.cut_high_k_off(filters_set[j1], dx1, dy1)
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
                                I1_f_small = self.cut_high_k_off(I1_f, dx2, dy2)
                                wavelet_f2 = self.cut_high_k_off(filters_set[j2], dx2, dy2)
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
    
    def scattering_cov(
        self, data, if_large_batch=False, C11_criteria=None, 
        use_ref=False, normalization='P00', remove_edge=False, 
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
            ref_P00 = self.ref_scattering_cov['P00']
            ref_P11 = self.ref_scattering_cov['P11']

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
        
        C01_iso = torch.zeros((N_image,J,J,L), dtype=data_f.dtype)
        P11_iso = torch.zeros((N_image,J,J,L), dtype=data.dtype)
        C11_pre_norm_iso = torch.zeros((N_image,J,J,J,L,L), dtype=data_f.dtype)
        C11_iso = torch.zeros((N_image,J,J,J,L,L), dtype=data_f.dtype)
        
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
        
        # calculate the covariance and correlations of the scattering fields
        # only use the low-k Fourier coefs when calculating large-j scattering coefs.
        for j3 in range(0,J):
            dx3, dy3 = self.get_dxdy(j3)
            I1_f_small = self.cut_high_k_off(I1_f, dx3, dy3)
            data_f_small = self.cut_high_k_off(data_f, dx3, dy3)
            wavelet_f3 = self.cut_high_k_off(filters_set[j3], dx3, dy3)
            _, M3, N3 = wavelet_f3.shape
            wavelet_f3_squared = wavelet_f3**2
            if remove_edge: 
                edge_mask = self.cut_high_k_off(self.edge_masks[j3], dx3, dy3)
                edge_mask = edge_mask / edge_mask.mean((-2,-1))[:,None,None]
            else: 
                edge_mask = 1
            # a normalization change due to the cutoff of frequency space
            fft_factor = 1 /(M3*N3) * (M3*N3/M/N)**2
            for j2 in range(0,j3+1):
                # [N_image,l2,l3,x,y]
                P11_temp = (
                    I1_f_small[:,j2].view(N_image,L,1,M3,N3).abs()**2 * 
                    wavelet_f3_squared.view(1,1,L,M3,N3) * edge_mask
                ).mean((-2,-1)) * fft_factor
                if use_ref:
                    if normalization=='P11':
                        norm_factor_C01 = (ref_P00[:,None,j3,:] * ref_P11[:,j2,j3,:,:])**0.5
                    if normalization=='P00':
                        norm_factor_C01 = (ref_P00[:,None,j3,:] * ref_P00[:,j2,:,None])**0.5
                else:
                    if normalization=='P11':
                        norm_factor_C01 = (P00[:,None,j3,:] * P11_temp)**0.5
                    if normalization=='P00':
                        norm_factor_C01 = (P00[:,None,j3,:] * P00[:,j2,:,None])**0.5
                C01[:,j2,j3,:,:] = (
                    data_f_small.view(N_image,1,1,M3,N3) * 
                    torch.conj(I1_f_small[:,j2].view(N_image,L,1,M3,N3)) *
                    wavelet_f3_squared.view(1,1,L,M3,N3)
                ).mean((-2,-1)) * fft_factor / norm_factor_C01
                for j1 in range(0, j2+1):
                    if eval(C11_criteria):
                        if not if_large_batch:
                            # [N_image,l1,l2,l3,x,y]
                            C11_pre_norm[:,j1,j2,j3,:,:,:] = (
                                I1_f_small[:,j1].view(N_image,L,1,1,M3,N3) * 
                                torch.conj(I1_f_small[:,j2].view(N_image,1,L,1,M3,N3)) *
                                wavelet_f3_squared.view(1,1,1,L,M3,N3)
                            ).mean((-2,-1)) * fft_factor
                        else:
                            for l1 in range(L):
                            # [N_image,l2,l3,x,y]
                                C11_pre_norm[:,j1,j2,j3,l1,:,:] = (
                                    I1_f_small[:,j1,l1].view(N_image,1,1,M3,N3) * 
                                    torch.conj(I1_f_small[:,j2].view(N_image,L,1,M3,N3)) *
                                    wavelet_f3_squared.view(1,1,L,M3,N3)
                                ).mean((-2,-1)) * fft_factor
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
            )**0.5
        if normalization=='P11':
            if use_ref: P = ref_P11
            else: P = P11
            #.view(N_image,J,1,J,L,1,L) * .view(N_image,1,J,J,1,L,L)
            C11 = C11_pre_norm / (
                P[:,:,None,:,:,None,:] * P[:,None,:,:,None,:,:]
            )**0.5
        # average over l1 to obtain simple isotropic statistics
        P00_iso = P00.mean(-1)
        S1_iso  = S1.mean(-1)
        for l1 in range(L):
            for l2 in range(L):
                C01_iso[...,(l2-l1)%L] += C01[...,l1,l2]
                P11_iso[...,(l2-l1)%L] += P11[...,l1,l2]
                for l3 in range(L):
                    C11_pre_norm_iso[...,(l2-l1)%L,(l3-l1)%L]+=C11_pre_norm[...,l1,l2,l3]
                    C11_iso[...,(l2-l1)%L,(l3-l1)%L] += C11[...,l1,l2,l3]
        C01_iso /= L; P11_iso /= L; C11_pre_norm_iso /= L; C11_iso /= L
        
        
        # get a single, flattened data vector for_synthesis
        select_and_index = self.get_index(J, L, normalization, C11_criteria)
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
            C01_iso[:,select_and_index['select_2_iso']].real, 
            C01_iso[:,select_and_index['select_2_iso']].imag, 
            C11_iso[:,select_and_index['select_3_iso']].real, 
            C11_iso[:,select_and_index['select_3_iso']].imag
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
        
        C01_iso = torch.zeros((N_image,4,J,J,L), dtype=data_a_f.dtype)
        P11_a_iso = torch.zeros((N_image,J,J,L), dtype=data_a.dtype)
        P11_b_iso = torch.zeros((N_image,J,J,L), dtype=data_a.dtype)
        C11_iso = torch.zeros((N_image,4,J,J,J,L,L), dtype=data_a_f.dtype)
        Corr11_iso= torch.zeros((N_image,4,J,J,J,L,L), dtype=data_a_f.dtype)
        
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
            I1_a_f_small = self.cut_high_k_off(I1_a_f, dx3, dy3)
            I1_b_f_small = self.cut_high_k_off(I1_b_f, dx3, dy3)
            data_a_f_small = self.cut_high_k_off(data_a_f, dx3, dy3)
            data_b_f_small = self.cut_high_k_off(data_b_f, dx3, dy3)
            wavelet_f3 = self.cut_high_k_off(filters_set[j3], dx3, dy3)
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
                    wavelet_f3_squared.view(1,1,L,M3,N3)
                ).mean((-2,-1)) * fft_factor / norm_factor_C01_a
                C01[:,1,j2,j3,:,:] = (
                    data_b_f_small.view(N_image,1,1,M3,N3) * 
                    torch.conj(I1_b_f_small[:,j2].view(N_image,L,1,M3,N3)) *
                    wavelet_f3_squared.view(1,1,L,M3,N3)
                ).mean((-2,-1)) * fft_factor / norm_factor_C01_b
                C01[:,2,j2,j3,:,:] = (
                    data_a_f_small.view(N_image,1,1,M3,N3) * 
                    torch.conj(I1_b_f_small[:,j2].view(N_image,L,1,M3,N3)) *
                    wavelet_f3_squared.view(1,1,L,M3,N3)
                ).mean((-2,-1)) * fft_factor / norm_factor_C01_ab
                C01[:,3,j2,j3,:,:] = (
                    data_b_f_small.view(N_image,1,1,M3,N3) * 
                    torch.conj(I1_a_f_small[:,j2].view(N_image,L,1,M3,N3)) *
                    wavelet_f3_squared.view(1,1,L,M3,N3)
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
                C01_iso[...,(l2-l1)%L] += C01[...,l1,l2]
                P11_a_iso[...,(l2-l1)%L] += P11_a[...,l1,l2]
                P11_b_iso[...,(l2-l1)%L] += P11_b[...,l1,l2]
                for l3 in range(L):
                    C11_iso   [...,(l2-l1)%L,(l3-l1)%L] +=    C11[...,l1,l2,l3]
                    Corr11_iso[...,(l2-l1)%L,(l3-l1)%L] += Corr11[...,l1,l2,l3]
        C01_iso /= L; P11_a_iso /= L; P11_b_iso /= L; C11_iso /= L; Corr11_iso /= L
        
        # generate single, flattened data vector for_synthesis
        select_and_index = self.get_index(J, L, normalization, C11_criteria, 2)
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
            C01_iso[:,:,select_and_index['select_2_iso']].reshape((N_image, -1)).real, 
            C01_iso[:,:,select_and_index['select_2_iso']].reshape((N_image, -1)).imag, 
            Corr11_iso[:,:,select_and_index['select_3_iso']].reshape((N_image, -1)).real, 
            Corr11_iso[:,:,select_and_index['select_3_iso']].reshape((N_image, -1)).imag
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

class Bispectrum_Calculator(object):
    def __init__(self, M, N, k_range=None, bins=None, bin_type='log', device='gpu'):
        if not torch.cuda.is_available(): device='cpu'
        # k_range in unit of pixel in Fourier space
        self.device = device
        if k_range is None:
            if bin_type=='linear':
                k_range = np.linspace(1, M/2*1.4, bins+1) # linear binning
            if bin_type=='log':
                k_range = np.logspace(0, np.log10(M/2*1.4), bins+1) # log binning
        self.k_range = k_range
        self.M = M
        self.N = N
        X, Y = np.meshgrid(np.arange(M), np.arange(N), indexing='ij')
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
                    if True: #i2 + i3 >= i1 :
                        self.select[i1, i2, i3] = True
                        self.B_ref_array[i1, i2, i3] = (refs[i1] * refs[i2] * refs[i3]).mean()
        if device=='gpu':
            self.k_filters_torch = self.k_filters_torch.cuda()
            self.select = self.select.cuda()
            self.B_ref_array = self.B_ref_array.cuda()
    
    def forward(self, image, normalization='image'):
        '''
        normalization is either 'image' or 'dirac'.
        '''
        if type(image) == np.ndarray:
            image = torch.from_numpy(image)

        B_array = torch.zeros(
            (len(image), len(self.k_range)-1, len(self.k_range)-1, len(self.k_range)-1), 
            dtype=image.dtype
        )
        
        if self.device=='gpu':
            image   = image.cuda()
            B_array = B_array.cuda()
        
        image_f = torch.fft.fftn(image, dim=(-2,-1))
        if normalization=='image':
            conv = torch.fft.ifftn(
                image_f[None,...] * self.k_filters_torch[:,None,...],
                dim=(-2,-1)
            ).real
        if normalization=='dirac':
            conv = torch.fft.ifftn(
                image_f[None,...]*0 + self.k_filters_torch[:,None,...],
                dim=(-2,-1)
            ).real
        conv_std = conv.std((-1,-2))
        for i1 in range(len(self.k_range)-1):
            for i2 in range(i1+1):
                for i3 in range(i2+1):
                    if True: #i2 + i3 >= i1 :
                        B = conv[i1] * conv[i2] * conv[i3]
                        B_array[:, i1, i2, i3] = B.mean((-2,-1)) / \
                            conv_std[i1] / conv_std[i2] / conv_std[i3]
                        # *1e8 # / self.B_ref_array[k1, k2, k3]
        return B_array.reshape(len(image), (len(self.k_range)-1)**3)[:,self.select.flatten()]

def get_power_spectrum(image, k_range=None, bins=None, bin_type='log', device='gpu'):
    '''
    get the power spectrum of a given image
    '''
    if type(image) == np.ndarray:
        image = torch.from_numpy(image)      
    if not torch.cuda.is_available(): device='cpu'
    if device=='gpu':
        image = image.cuda()
            
    M, N = image.shape[-2:]
    modulus = torch.fft.fftn(image, dim=(-2,-1)).abs()
    
    modulus = torch.cat(
        ( torch.cat(( modulus[..., M//2:, N//2:], modulus[..., :M//2, N//2:] ), -2),
          torch.cat(( modulus[..., M//2:, :N//2], modulus[..., :M//2, :N//2] ), -2)
        ),-1)
    
    X = torch.arange(0,M)
    Y = torch.arange(0,N)
    Ygrid, Xgrid = torch.meshgrid(Y,X, indexing='ij')
    k = ((Xgrid - M/2)**2 + (Ygrid - N/2)**2)**0.5
    
    if k_range is None:
        if bin_type=='linear':
            k_range = torch.linspace(1, M/2*1.4, bins+1) # linear binning
        if bin_type=='log':
            k_range = torch.logspace(0, np.log10(M/2*1.4), bins+1) # log binning

    power_spectrum = torch.zeros(len(image), len(k_range)-1, dtype=image.dtype)
    if device=='gpu':
        k = k.cuda()
        k_range = k_range.cuda()
        power_spectrum = power_spectrum.cuda()

    for i in range(len(k_range)-1):
        select = (k >= k_range[i]) * (k < k_range[i+1])
        power_spectrum[:,i] = ((modulus**2*select[None,...]).sum((-2,-1))/select.sum()).log()
    return power_spectrum, k_range

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


import torch.fft as fft
import scipy.io as sio
import math
import torch.nn.functional as F
from scattering.backend import cdgmm, Modulus, SubsampleFourier, Pad, mulcu, \
    SubInitSpatialMeanR, SubInitSpatialMeanC, DivInitStdR, DivInitStd, \
    padc, conjugate, maskns, masks_subsample_shift3, \
    extract_shift3

class AlphaScattering2d_cov(object):
    def __init__(
        self, M, N, J, L=4, A=4, A_prime=1, delta_j=1, delta_l=4,
        nb_chunks=1, chunk_id=0, shift='all', wavelets='morlet',
        filter_path=None,#'./filters/'
        device='gpu',
    ):
        if not torch.cuda.is_available(): device='cpu'
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
        self.device = device
        # print(self.wavelets)
        assert(self.chunk_id <= self.nb_chunks)
        if self.delta_l > self.L:
            raise (
                ValueError('delta_l must be <= L'))
        self.build()
        # move variables into GPU
        if self.device=='gpu':
            self.cuda()

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
            self.filters_tensor_bump()
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
            hatpsi_ = torch.load(self.path+'morlet_N'+str(N)+'_J'+str(J)+'_L'+str(L)+'.pt') # (J,L,M,N,2) torch
            hatpsi = torch.cat((hatpsi_, torch.flip(hatpsi_, (2,3))), dim=1).numpy() # (J,L2,M,N,2) numpy
            fftpsi = hatpsi[...,0] + hatpsi[...,1]* 1.0j # numpy
            hatphi = torch.load(self.path+'/morlet_lp_N'+str(N)+'_J'+str(J)+'_L'+str(L)+'.pt').numpy()  # (M,N,2) numpy
        else:
            Sihao_filters = FiltersSet(M=N, N=N, J=J, L=L).generate_morlet()
            hatpsi_ = Sihao_filters['psi'] # (J,L,M,N)
            hatpsi_ = torch.cat((hatpsi_[...,None], hatpsi_[...,None]*0), dim=-1) # (J,L,M,N,2) torch
            hatpsi = torch.cat((hatpsi_, torch.flip(hatpsi_, (2,3))), dim=1).numpy() # (J,L2,M,N,2) numpy
            fftpsi = hatpsi[...,0] + hatpsi[...,1]* 1.0j # numpy
            hatphi = Sihao_filters['phi']  # (M,N)
            hatphi = torch.cat((hatphi[...,None], hatphi[...,None]*0), dim=-1).numpy() # (M,N,2) numpy

        
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

    def filters_tensor_bump(self):

        J = self.J
        M = self.M; N = self.N; L = self.L
        # load phi filters

        assert(self.M == self.N)
        if self.path is not None:
            matfilters = sio.loadmat(self.path+'matlab/filters/bumpsteerableg1_fft2d_N'
                                     + str(self.N) + '_J' + str(self.J) + '_L'
                                     + str(self.L) + '.mat')
    
            fftphi = matfilters['filt_fftphi'].astype(np.complex_)  # (M,N) numpy
            hatphi = np.stack((np.real(fftphi), np.imag(fftphi)), axis=-1) # numpy
    
            fftpsi = matfilters['filt_fftpsi'].astype(np.complex_)  # (J,L2,M,N)
            # hatpsi = np.stack((np.real(fftpsi), np.imag(fftpsi)), axis=-1)
        else:
            Sihao_filters = FiltersSet(M=N, N=N, J=J, L=L).generate_bump_steerable()
            hatpsi_ = Sihao_filters['psi'] # (J,L,M,N)
            hatpsi_ = torch.cat((hatpsi_[...,None], hatpsi_[...,None]*0), dim=-1) # (J,L,M,N,2) torch
            hatpsi = torch.cat((hatpsi_, torch.flip(hatpsi_, (2,3))), dim=1).numpy() # (J,L2,M,N,2) numpy
            fftpsi = hatpsi[...,0] + hatpsi[...,1]* 1.0j # numpy
            hatphi = Sihao_filters['phi']  # (M,N)
            hatphi = torch.cat((hatphi[...,None], hatphi[...,None]*0), dim=-1).numpy() # (M,N,2) numpy



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

        # self.hatphi = torch.FloatTensor(hatphi)  # (M,N,2)
        # self.hatpsi = torch.FloatTensor(filters)
        # self.hatpsi_prime = torch.FloatTensor(filters_prime)
        self.hatphi = torch.view_as_complex(torch.FloatTensor(hatphi)).type(torch.cfloat)  # (M,N,2)
        self.hatpsi = torch.view_as_complex(torch.FloatTensor(filters)).type(torch.cfloat)
        self.hatpsi_prime = torch.view_as_complex(torch.FloatTensor(filters_prime)).type(torch.cfloat)

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
        
        # convert numpy array input into torch tensors
        if type(input) == np.ndarray:
            input = torch.from_numpy(input)
        if self.device == 'gpu':
            input = input.cuda()
        
        x_c = padc(input)  # add zeros to imag part -> (nb,M,N)
        hatx_c = fft.fft2(torch.view_as_complex(x_c)).type(torch.cfloat)  # fft2 -> (nb,M,N)
        
        if self.chunk_id < self.nb_chunks:
            nb = hatx_c.shape[0]
            hatpsi_la = self.hatpsi[:,:L,...]  # (J,L,A,M,N)
            nb_channels = self.this_wph['la1'].shape[0]
            t = 3 if wavelets == 'morlet' else 1 if wavelets == 'bump' else 0
            if self.chunk_id < self.nb_chunks-1:
                Sout = input.new(nb,nb_channels,M,N)
            else:
                Sout = input.new(nb,nb_channels+1+t,M,N)
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
            Sout = Sout.view(nb, -1)
            Sout = torch.cat((Sout, input.mean((-2,-1)).view(nb,1), input.std((-2,-1)).view(nb,1)), dim=1)*1e-4
        return Sout

    def __call__(self, input):
        return self.forward(input)
