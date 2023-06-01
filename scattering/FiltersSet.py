import numpy as np
import torch
import torch.fft

class FiltersSet(object):
    def __init__(self, M, N, J=None, L=4):
        if J is None:
            J = int(np.log2(min(M,N))) - 1
        self.M = M
        self.N = N
        self.J = J
        self.L = L
        
    def generate_wavelets(
        self, if_save=False, save_dir=None, 
        wavelets='morlet', precision='single', 
        l_oversampling=1, frequency_factor=1
    ):
        # Morlet Wavelets
        if precision=='double':
            dtype = torch.float64
            dtype_np = np.float64
        if precision=='single':
            dtype = torch.float32
            dtype_np = np.float32
        if precision=='half':
            dtype = torch.float16
            dtype_np = np.float16

        psi = torch.zeros((self.J, self.L, self.M, self.N), dtype=dtype)
        for j in range(self.J):
            for l in range(self.L):
                k0 = frequency_factor * 3.0 / 4.0 * np.pi /2**j
                theta0 = (int(self.L-self.L/2-1)-l) * np.pi / self.L
                
                if wavelets=='morlet':
                    wavelet_spatial = self.morlet_2d(
                        M=self.M, N=self.N, xi=k0, theta=theta0,
                        sigma=0.8 * 2**j / frequency_factor, 
                        slant=4.0 / self.L * l_oversampling,
                    )
                    wavelet_Fourier = np.fft.fft2(wavelet_spatial)
                if wavelets=='BS':
                    wavelet_Fourier = self.bump_steerable_2d(
                        M=self.M, N=self.N, k0=k0, theta0=theta0,
                        L=self.L
                    )
                if wavelets=='gau':
                    wavelet_Fourier = self.gau_steerable_2d(
                        M=self.M, N=self.N, k0=k0, theta0=theta0,
                        L=self.L
                    )
                if wavelets=='shannon':
                    wavelet_Fourier = self.shannon_2d(
                        M=self.M, N=self.N, kmin=k0 / 2**0.5, kmax=k0 * 2**0.5, theta0=theta0,
                        L=self.L
                    )
                wavelet_Fourier[0,0] = 0
                psi[j, l] = torch.from_numpy(wavelet_Fourier.real.astype(dtype_np))
                    
        if wavelets=='morlet':
            phi = torch.from_numpy(
                self.gabor_2d_mycode(
                    self.M, self.N, 0.8 * 2**(self.J-1) / frequency_factor, 0, 0
                ).real.astype(dtype_np)
            ) * (self.M * self.N)**0.5
        if wavelets=='BS':
            phi = torch.from_numpy(
                self.gabor_2d_mycode(
                    self.M, self.N, 2 * np.pi /(0.702*2**(-0.05)) * 2**(self.J-1) / frequency_factor, 0, 0
                ).real.astype(dtype_np)
            ) * (self.M * self.N)**0.5
        if wavelets=='gau':
            phi = torch.from_numpy(
                self.gabor_2d_mycode(
                    self.M, self.N, 2 * np.pi /(0.702*2**(-0.05)) * 2**(self.J-1) / frequency_factor, 0, 0
                ).real.astype(dtype_np)
            ) * (self.M * self.N)**0.5
        if wavelets=='shannon':
            phi = torch.from_numpy(
                self.shannon_2d(
                    M=self.M, N=self.N, kmin = -1, 
                    kmax = frequency_factor * 0.375 * 2 * np.pi / 2**self.J * 2**0.5,
                    theta0 = 0, L = 0.5
                )
            )
            
        filters_set = {'psi':psi, 'phi':phi}
        if if_save:
            np.save(
                save_dir + 'filters_set_M' + str(self.M) + 'N' + str(self.N)
                + 'J' + str(self.J) + 'L' + str(self.L) + '_' + precision + '.npy', 
                np.array([{'filters_set': filters_set}])
            )
        return filters_set
    
    # Morlet Wavelets
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
        K = wv.sum() / wv_modulus.sum()

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
        c = 1/1.29 * 2**(L/2-1) * np.math.factorial(int(L/2-1)) / np.sqrt((L/2)* np.math.factorial(int(L-2)))
        bump_steerable_fft = c * (radial * angular).sum((0,1))
        return bump_steerable_fft

    # Gaussian Steerable Wavelet
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
        c = 1/1.29 * 2**(L/2-1) * np.math.factorial(int(L/2-1)) / np.sqrt((L/2)* np.math.factorial(int(L-2)))
        gau_steerable_fft = c * (radial * angular).sum((0,1))
        return gau_steerable_fft

    # tophat (Shannon) Wavelet
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
    def generate_gau_harmonic(self, if_save=False, save_dir=None, precision='single', frequency_factor=1):
        if precision=='double':
            psi = torch.zeros((self.J, self.L, self.M, self.N), dtype=torch.complex128)
        if precision=='single':
            psi = torch.zeros((self.J, self.L, self.M, self.N), dtype=torch.complex64)
        # psi
        for j in range(self.J):
            for l in range(self.L):
                wavelet_Fourier = self.gau_harmonic_2d(
                    M=self.M, N=self.N, k0= frequency_factor * 0.375 * 2 * np.pi / 2**j, l=l,
                )
                wavelet_Fourier[0,0] = 0
                if precision=='double':
                    psi[j,l] = torch.from_numpy(wavelet_Fourier)
                if precision=='single':
                    psi[j,l] = torch.from_numpy(wavelet_Fourier.astype(np.complex64))
        # phi
        if precision=='double':
            phi = torch.from_numpy(
                self.gabor_2d_mycode(
                    self.M, self.N, 2 * np.pi /(0.702*2**(-0.05)) * 2**(self.J-1) / frequency_factor, 0, 0
                ).real.astype(np.float64)
            ) * (self.M * self.N)**0.5
        if precision=='single':
            phi = torch.from_numpy(
                self.gabor_2d_mycode(
                    self.M, self.N, 2 * np.pi /(0.702*2**(-0.05)) * 2**(self.J-1) / frequency_factor, 0, 0
                ).real.astype(np.float32)
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
