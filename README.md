# The `scattering` package

The scattering transform provides a powerful statistical vocabulary to quantify textures in a signal / field. It is similar to the power spectrum, but it captures a lot more information about complex non-Gaussian structures which are ubiquitous in astronomical/physical data.

In this python3 package called "scattering", we provide a set of functions to calculate the scattering coefficients, scattering covariance, alpha covariance, and binned bispectrum, with similar format. This package allows one to perform both analysis and synthesis within the same framework. The input images can be a numpy array or torch tensor with size `[N_image, N_x_pixels, N_y_pixels]`. Below we show example usage, which can also be found in the demo jupyter notebook `scattering.ipynb` in this repository.

## Install
Please download (only) the folder `scattering` to one of the system paths. Or, you can download it to any folder and add that folder to system paths: 
```python
import sys
sys.path.append('~/where/you/download/the/script/')
``` 
Then, simply import it:
```python
import scattering
```

## Analysis

### modulus scattering
1. define calculator
```python
st_calc = scattering.Scattering2d(M=256, N=256, J=5, L=4)
```
2. calculate the scattering coefficients (scattering mean) and scattering covariance:
```python
s_mean = st_calc.scattering_coef(image_input)
s_cov  = st_calc.scattering_cov (image_input)

print(s_mean['S2'])
print(s_cov['C11_iso'])
```

### alpha scattering

1. define calculator:
```python
aw_calc = scattering.AlphaScattering2d_cov(M=256, N=256, J=5, L=4)
```

2. calculate the alpha correlations:
```python
alpha_cov = aw_calc.forward(image_input)
print(alpha_cov)
```

### bispectrum

1. define calculator:
```python
k_bins = 5
M = N = 256
k_range = np.logspace(0,np.log10(M/2*1.4), k_bins)
bi_calc = scattering.Bispectrum_Calculator(k_range, M=M, N=N)
```

2. calculate binned bispectrum
```python
bi = bi_calc.forward(image_input)
```

### using gpu / cpu

To save computation time, the default device is gpu. However, if no gpu is found in the system, the package will automatically switch to cpu. If you have gpu access but still want to enforce the use of cpu, please set parameter `device='cpu'` when defining the calculator.
```python
st_calc = scattering.Scattering2d(M=256, N=256, J=5, L=4, device='cpu')
```


## Synthesis example

We provide a simple function to perform image synthesis based on the aforementioned coefficients. The logic of using gpu/cpu is the same as described above.

### generating new images with similar textures an/some target image(s)
```python
image_syn = scattering.synthesis(estimator_name='s_cov_iso', target=image_input, mode='image')
```

This is an example synthesis result based on the scattering covariance. The left panel is the target image and the right is the synthesised one.

![](https://github.com/abrochar/wavelet-ops/blob/main/synthesis_image.png?raw=true)


### generating new images with target values for particular coefficients

```python
image_syn = scattering.synthesis(
    estimator_name='s_cov_iso', 
    target=coef_target,
    mode='estimator', 
    M=256, N=256, J=5, L=4,
    steps=400, learning_rate=0.5
)
```

This is an example of interpolating the scattering covariance values from two fields. The leftmost and rightmost ones are two input images.

![](https://github.com/abrochar/wavelet-ops/blob/main/synthesis_coef.png?raw=true)









# More details about the `ST.py` module

Inside the package of `scattering` there is a module called `ST.py`, which is a python3 module to calculate the scattering mean and covariance coefficients of 1D signals or 2D fields (images), and can be used independently from the scattering package. It has been optimized in speed, convenience, and flexibility. Everything you need is just one python script `ST.py`, which depends only on two packages: `numpy, torch = 1.7+`. 

This `ST.py` module can do the following:
1. Creating the 1D or 2D wavelets to be used in scattering transform;
2. Calculating the scattering coefficients of 1D signals or 2D fields (images).
3. Calculating the phase harmonic correlations, a statistic similar to the scattering transform but enabling non-linear cross-correlation between two signal / fields.
The code has been optimized for both speed and memory use. Code for 3D cases are in working progress.

For questions or suggestions or comments, please do not hesitate to contact me: s.cheng@jhu.edu

## Install
Please download (only) the script `ST.py` to one of the system paths. Or, you can download it to any folder and add that folder to system paths: 
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
1. It has a fast algorithm for the global scattering coefficients, which can speed up about 5 times depending on the image and batch sizes. For example, for a large batch of 256x256 images, J=7 and L=4, the speed is around 1000 images per second (using a Tesla P100 GPU with google colab) and 30 images per second (using one CPU), which is 4 and 10 times faster than using the kymatio pytorch backend for the same settings.
2. It is compact and easy-to-modify.
3. It allows for customized wavelet set.
4. It generates wavelet filter bank much faster, due to a simple optimization in the code.

The advantages of `kymatio` package are:
1. It allows for calculating local scattering coefficients.
2. It also contains codes for 3D applications.
(I am working on adding these functions to my code. Also, part of my code for generating the Morlet wavelets was copied from the `kymatio` package.)

## Example 1: computing scattering transform

Here I show the basic usage. First, generate the Morlet wavelets to be used.
```python
J = 8
L = 4
M = 512
N = 512
filter_set = ST.FiltersSet(M, N, J, L).generate_morlet(precision='single')
```
Other wavelets are also possible. For example, the function "generate_bump_steerable" will generate bump steerable wavelets. You may also use your own filters by slightly modifying the code.

Then, define a ST calculator and feed it with images:
```python
ST_calculator = ST.ST_2D(filter_set, J, L, device='gpu', weight=None)

input_images = np.empty((30, M, N), dtype=np.float32)

S, S0, S1, S2, _, _, _, _ = ST_calculator.forward(input_images, J, L)
```

The input data should be a numpy array or torch tensor of images with dimensions (N_image, M, N). Output are torch tensors in the assigned computing device, e.g., cuda() or cpu. Parallel calculation is automatically implemented by `torch`, for both cpu and gpu. The code is optimized to have as much parallel computation as possible, but when the number of images in a batch is large, setting "if_large_batch=True" will significantly reduce the memory use (in this large-batch case, as the parallelization is already assigned among images, this setting will not reduce the speed. But in small-batch cases, please just use the default if_large_batch=False).
```python
S, S0, S1, S2, _, _, _, _ = ST_calculator.forward(
    input_images, J, L, if_large_batch=True
)
```

When using CPUs, one may set torch.set_num_threads(int) to the number of CPUs. Then, pytorch will deal with the parallelization by itself.

S has dimension (N_image, 1 + J + JxJxL), which keeps the (l1-l2) dimension.

S_0 has dimension (N_image, 1)

S_1 has dimension (N_image, J, L)

S_2 has dimension (N_image, J, L, J, L)

The default j1j2_criteria='j2>j1' means that only coefficients with j2>j1 are calculated. Uncalculated
coefficients are set to zeros in the output arrays.


## Example 2: computing alpha-phase correlations

Similarly, I also provide a fast code for computing a subset of alpha-phase correlations. Again, first we can generate the Morlet wavelets to be used.
```python
J = 8
L = 4
M = N = 512
filter_set = ST.FiltersSet(M, N, J, L).generate_morlet(precision='single')
```
Then, define a ST calculator and feed it with images. To compute the alpha-phase correlations, we call the method "phase_harmonics" instead of "forward".
```python
ST_calculator = ST.ST_2D(filter_set, J, L, device='gpu', weight=None)

input_images = np.empty((30, M, N), dtype=np.float32)

PH = ST_calculator.phase_harmonics(
    input_images, J, L
)
```

It returns a dictionary with different sets of coefficients including three types of alpha-phase correlations. (Note that the notation here is slightly different from the original paper. Here we use the number to represent the order of non-linearity. So C00 are the linear correlations, C01 are the correlation between the original field and modulus field (0th- and 1st-order non-linear fields), etc.)
| type    | definition |
| ------  | ----------- |
|orig. x orig.    |   C00 = <(I * psi)(I * psi)>  |
|orig. x modulus  |   C01 = <(I * psi2)(\|I * psi1\| * psi2)> / sqrt(\|\|I * psi2\|\| x \|\| \|I * psi1\| * psi2 \|\|)  |
|modulus x modulus|  C11 = <(|I * psi1| * psi3)(|I * psi2| * psi3)>   |


In particular:
| key    | Description |
| ------ | ----------- |
| 'C00'  | torch tensor with size [N_image, J, L], the power in each wavelet bands      |
| 'S1'   | torch tensor with size [N_image, J, L] the 1st-order scattering coefficients, i.e., the mean of wavelet modulus fields        |
|'C01_iso' | torch tensor with size [N_image, JxJxL], the orig. x modulus terms averaged over l1. It is flattened from a tensor of size [N_image, J, J, L], where the elements not following j1 < j2 are all set to zeros.
|'P11'| torch tensor with size [N_image, J, J, L, L], the modulus x modulus terms with j1=j2 and l1=l2. Elements not following j1 < j3 are all set to np.nan.
|'P11_iso'| torch tensor with size [N_image, J, J, L], the modulus x modulus terms with j1=j2 and l1=l2, averaged over l1. Elements not following j1 < j3 are all set to np.nan.
|'C11' | torch tensor with size [N_image, J, J, J, L, L, L], the modulus x modulus terms in general. Elements not following j1 <= j2 < j3 are all set to np.nan.
|'C11_iso' | torch tensor with size [N_image, J, J, J, L, L], the modulus x modulus terms in general. Elements not following j1 <= j2 < j3 are all set to np.nan.


## Example 3: image synthesis with gradient descent

Please refer to the two jupyter notebooks "ST-image-synthesis" and "PH-image-synthesis".
