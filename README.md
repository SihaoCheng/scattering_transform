# scattering transform (ST)

The scattering transform provides a powerful statistical vocabulary to quantify textures in a signal / field. It is similar to the power spectrum, but it captures a lot more information, particularly about non-Gaussian textures which are ubiquitous in astronomical/physical data.

Here I provide a python3 module to calculate the scattering coefficients of 2D fields (images). It has been optimized in speed, convenience, and flexibility. Everything you need is just one python script `ST.py`, which depends only on two packages: `numpy, torch = 1.7+`. 

This `ST` module can do the following:
1. Creating the 2D wavelets to be used in scattering transform;
2. Calculating the scattering coefficients of 2D fields (images).
Codes for 1D or 3D cases are in working progress.

For questions or suggestions or comments, please do not hesitate to contact me: s.cheng@jhu.edu

## Install
Please download the script `ST.py` to one of the system paths. Or, you can download it to any folder and add that folder to system paths: 
```python
import sys
sys.path.append('~/where/you/download/the/script/')
``` 
Then, simply import it:
```python
import ST
```

## Comparison to `kymatio`

There is another python package called `kymatio` also dedicated to the scattering transform. The two modules are similar in general -- both of them can do:
1. calculate scattering coefficients for a batch of images;
2. switching between CPU/GPU calculation.

However, there are several practical differences. The advantages of my `ST.py` module are:
1. I provide an option of using a fast algorithm, which can speed up about 5 times (depending on the size of image);
2. It is compact and easy-to-modify.
3. It allows for customized wavelet set.
4. It uses pytorch >= 1.7, which is better optimized for FFT. 
5. It generates wavelets much faster, with a small optimization.

The advantages of `kymatio` package are:
1. It allows for calculating local scattering coefficients.
2. It also contains codes for 1D and 3D applications.
(I am working on adding these functions to my code. Also, part of my code for generating the Morlet wavelets was copied from the `kymatio` package.)

## Example 1

Here I show the basic usage. First, generate the Morlet wavelets to be used.
```python
J = 8
L = 4
M = 512
N = 512

save_dir = '#####'
filter_set = ST.FiltersSet(M, N, J, L).generate_morlet(
    if_save=True, save_dir=save_dir, precision='single'
)
```
You may choose to save these wavelets and load them later:
```python
filter_set = np.load(
    save_dir + 'filters_set_M' + str(M) + 'N' + str(N) + 
    'J' + str(J) + 'L' + str(L) + '_single.npy',
    allow_pickle=True
)[0]['filters_set']
```
Then, define a ST calculator and feed it with images:
```python
ST_calculator = ST.ST_2D(filter_set, J, L, device='gpu', weight=None)

input_images = np.empty((30, M, N), dtype=np.float32)

S, S_0, S_1, S_2 = ST_calculator.forward(
    input_images, J, L, algorithm='fast'
)
```

The input data should be a numpy array or torch tensor of images with dimensions (N_image, M, N). Output are torch tensors in the assigned computing device, e.g., cuda() or cpu. Parallel calculation is automatically implemented by `torch`, for both cpu and gpu. Please pay attention that large number of images in a batch (30 in this example) may cause memory problem. In that case just cut the image set into smaller batches. 

When using CPUs, one may also consider to feed 1 image (with size [1, M, N]) in each batch, and use the 'multiprocessing' package for parallel computation.

S has dimension (N_image, 1 + J + JxJxL), which keeps the (l1-l2) dimension.

S_0 has dimension (N_image, 1)

S_1 has dimension (N_image, J, L)

S_2 has dimension (N_image, J, L, J, L)

E has dimension (N_image, J, L), it is the power in each 1st-order wavelet bands

E_residual has dimension (N_image, J, L, J, L), it is the residual power in the 2nd-order scattering fields, which is the residual power not extracted by the scattering coefficients.

j1j2_criteria='j2>j1' assigns which S2 coefficients to calculate. Uncalculated
coefficients will have values of zero.

