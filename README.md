# scattering transform (ST)

The scattering transform provides a powerful statistical vocabulary to quantify textures in a signal / field. In a sense, it is similar to the power spectrum, but it captures a lot more information than the power spectrum, particularly about non-Gaussian textures, which are ubiquitous in astronomical/physical data.

Here I provide a python 3 module to calculate the scattering coefficients of 2D fields (images). The module `ST.py` depends only on the following packages: 
`numpy, torch >= 1.7`. 

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
4. It uses pytorch >= 1.7, which is better optimized. My code for generating wavelets is also faster.

The advantages of `kymatio` package are:
1. It allows for calculating local scattering coefficients.
2. It also contains codes for 1D and 3D applications.

Part of my code for generating the Morlet wavelets was copied from the `kymatio` package.

## Examples

Below, I show some example usage.
