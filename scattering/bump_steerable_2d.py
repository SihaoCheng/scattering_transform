import numpy as np

# Bump Steerable Wavelet
def bump_steerable_2d(M, N, k0, theta0, L):
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
