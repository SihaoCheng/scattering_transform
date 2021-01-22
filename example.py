import numpy as np
import torch
import ST_Jan22

J = 8
L = 4
M = 512
N = 512

f = ST.FiltersSet(M, N, J, L)
f.generate_morlet(if_save=True, save_dir='', precision='single')
filters_set = np.load('filters_set_mycode_M'+str(M)+
    'N'+str(N)+'J'+str(J)+'L'+str(L)+'_single.npy',
    allow_pickle=True)[0]['filters_set']
ST_calculator = ST.ST_mycode_new(filters_set, J, L, device='gpu')

############ DEFINE DATA ARRAY #########
data = np.empty((20, M, N))

################## ST ##################
S, S_0, S_1, S_2 = ST_calculator.forward(
  data, J, L, j1j2_criteria='j2>j1', algorithm='fast')
