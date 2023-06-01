import numpy as np
import torch
import torch.fft

if __name__ == "__main__":
    get_power_spectrum()

# power spectrum computer
def get_power_spectrum(image, k_range=None, bins=None, bin_type='log', device='gpu'):
    '''
    get the power spectrum of a given image
    '''
    if type(image) == np.ndarray:
        image = torch.from_numpy(image)    
    if type(k_range) == np.ndarray:
        k_range = torch.from_numpy(k_range) 
    if not torch.cuda.is_available(): device='cpu'
    if device=='gpu':
        image = image.cuda()
            
    M, N = image.shape[-2:]
    modulus = torch.fft.fftn(image, dim=(-2,-1), norm='ortho').abs()
    
    modulus = torch.cat(
        ( torch.cat(( modulus[..., M//2:, N//2:], modulus[..., :M//2, N//2:] ), -2),
          torch.cat(( modulus[..., M//2:, :N//2], modulus[..., :M//2, :N//2] ), -2)
        ),-1)
    
    X = torch.arange(M)[:,None]
    Y = torch.arange(N)[None,:]
    Xgrid = X+Y*0
    Ygrid = X*0+Y
    k = ((Xgrid - M/2)**2 + (Ygrid - N/2)**2)**0.5
    
    if k_range is None:
        if bin_type=='linear':
            k_range = torch.linspace(1, M/2*1.415, bins+1) # linear binning
        if bin_type=='log':
            k_range = torch.logspace(0, np.log10(M/2*1.415), bins+1) # log binning

    power_spectrum = torch.zeros(len(image), len(k_range)-1, dtype=image.dtype)
    if device=='gpu':
        k = k.cuda()
        k_range = k_range.cuda()
        power_spectrum = power_spectrum.cuda()

    for i in range(len(k_range)-1):
        select = (k > k_range[i]) * (k <= k_range[i+1])
        power_spectrum[:,i] = ((modulus**2*select[None,...]).sum((-2,-1))/select.sum()).log()
    return power_spectrum, k_range
