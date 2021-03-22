# scattering transform (ST)

The scattering transform provides a powerful statistical vocabulary to quantify textures in a signal / field. In a sense, it is similar to the power spectrum, but it captures a lot more information than the power spectrum, particularly about non-Gaussian textures, which are ubiquitous in astronomical/physical data.

Here I provide a python 3 module to calculate the scattering coefficients of 2D fields (images). The module `ST.py` depends only on the following packages: 
`numpy, torch = 1.7`. 

It can do the following things:
1. Create the 2D wavelets to be used in scattering transform;
2. Calculate the scattering coefficients of 2D fields (images).
The codes for 1D or 3D cases are in working progress.

For questions or suggestions or comments, please do not hesitate to contact me: s.cheng@jhu.edu

## Install
Please download the script `ST.py` to one of the system paths of your python, and then simply import it in python 3:
```python
import ST
```
Of course, you can add any paths to the system paths. Just add
```python
import sys
sys.path.append('~/where/you/download/the/script/')
print(sys.path)
```
before importing `ST`.

## Comparison to `kymatio`

There is another python package called `kymatio`, which is also dedicated to performing the scattering transform and is much more formal. The two modules are similar in general. Both my `ST.py` module and the `kymatio` package can do:
1. calculate scattering coefficients for a batch of image;
2. switching between CPU/GPU calculation.

However, there are also several practical differences. The advantages of my `ST.py` module are:
1. When the image size is dyadic, such as 256x128 pixels, I provide a fast algorithm with a speeds-up factor of 5;
2. It is compact and easy-to-modify.
3. It allows for customized wavelet set.
4. It uses pytorch = 1.7, which is better optimized for FFT. My code for generating wavelets is also faster.

The advantages of `kymatio` package are:
1. It allows for calculating local scattering coefficients.
2. It also contains codes for 1D and 3D applications.

Part of my code for generating the Morlet wavelets was copied from the `kymatio` package.

## Example 1

Here I show the basic usage. First generate the Morlet wavelets to be used.
```python
J = 8
L = 4
M = 512
N = 512

filter_set = ST.FiltersSet(M, N, J, L)
```
You may choose to save these wavelets:
```python
save_dir = '#####'
filter_set.generate_morlet(
    if_save=True, save_dir=save_dir, precision='single'
)
```
To load filters,
```python
filters_set = np.load(
    save_dir + 'filters_set_M' + str(M) + 'N' + str(N) + 
    'J' + str(J) + 'L' + str(L) + '_single.npy',
    allow_pickle=True
)[0]['filters_set']
```
Then, define a ST calculator, obtain dataset, and feed them to the calculator:
```python
ST_calculator = ST.ST_2D(filters_set, J, L, device='gpu')

input_image = np.empty((30, M, N), dtype=np.float32)

S, S_0, S_1, S_2 = ST_calculator.forward(
    input_images, J, L, 
    j1j2_criteria='j2>j1', algorithm='fast'
)

```

The input data should be a numpy array of images with dimensions (N_image, M, N). Output are torch tensors in assigned computing device, e.g., cuda() or cpu. Parallele calculation is automatically implemented by `torch`, for both cpu and gpu. Please pay attention that large number of images in a batch (30 in this example) may cause memory problem. In that case just cut it into smaller batchs. When using CPUs, one may also consider to feed 1 image in each batch, and use the 'multiprocessing' package for parallel computation.

S has dimension (N_image, 1 + J + JxJxL), which keeps the (l1-l2) dimension.

S_0 has dimension (N_image, 1)

S_1 has dimension (N_image, J, L)

S_2 has dimension (N_image, J, L, J, L)

j1j2_criteria='j2>j1' assigns which S2 coefficients to calculate. Uncalculated
coefficients will have values of zero.

