import numpy as np
import torch
import ST_Jan22

J = 8
L = 4
M = 512
N = 512

f = ST_Jan22.FiltersSet(M, N, J, L)

# generate and save morlet filter bank. "single" means single precision
save_dir = '#####'
f.generate_morlet(if_save=True, save_dir=save_dir, precision='single')

# load filter bank
filters_set = np.load(save_dir + 'filters_set_mycode_M' + str(M) + 
    'N' + str(N) + 'J' + str(J) + 'L' + str(L) + '_single.npy',
    allow_pickle=True)[0]['filters_set']

# define ST calculator
ST_calculator = ST_Jan22.ST_mycode_new(filters_set, J, L, device='gpu')

############ DEFINE DATA ARRAY #########
data = np.empty((30, M, N), dtype=np.float32)

################## ST ##################
# input data should be a numpy array of images with dimensions (N_image, M, N)
# output are torch tensors with assigned computing device, e.g., cuda() or cpu
# S has dimension (N_image, 1+J+J*J*L), which keeps the (l1-l2) dimension
# S_0 has dimension (N_image, 1)
# S_1 has dimension (N_image, J, L)
# S_2 has dimension (N_image, J, L, J, L)
# j1j2_criteria='j2>j1' assigns which S2 coefficients to calculate. Uncalculated
# coefficients will have values of zero.

S, S_0, S_1, S_2 = ST_calculator.forward(
  data, J, L, j1j2_criteria='j2>j1', algorithm='fast')
