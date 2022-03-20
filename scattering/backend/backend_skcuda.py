# Authors: Edouard Oyallon, Sergey Zagoruyko

from collections import defaultdict, namedtuple
import torch
from skcuda import cublas
import cupy
from string import Template

import torch.nn as nn
from torch.nn import ReflectionPad2d
from torch.autograd import Function
import numpy as np

NAME = 'skcuda'

from .backend_common import iscomplex # , real, imag # , mul


@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


Stream = namedtuple('Stream', ['ptr'])


def getDtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


class cdgmmMul(Function):
    @staticmethod
    def forward(ctx, A, B):
        """
        Complex pointwise multiplication between (batched) tensor A and tensor B.

        Parameters
        ----------
        A : tensor
            input tensor with size (B, C, M, N, 2)
        B : tensor
            B is a complex tensor of size (M, N, 2)
        inplace : boolean, optional
            if set to True, all the operations are performed inplace

        Returns
        -------
        C : tensor
            output tensor of size (B, C, M, N, 2) such that:
            C[b, c, m, n, :] = A[b, c, m, n, :] * B[m, n, :]
        """
        A, B = A.contiguous(), B.contiguous()
        ctx.save_for_backward(A,B)
        
        if A.size()[-3:] != B.size():
            raise RuntimeError('The filters are not compatible for multiplication!')
        
        if not iscomplex(A) or not iscomplex(B):
            raise TypeError('The input, filter and output should be complex')

        if B.ndimension() != 3:
            raise RuntimeError('The filters must be simply a complex array!')

        if type(A) is not type(B):
            raise RuntimeError('A and B should be same type!')

        if not A.is_cuda:
            raise RuntimeError('Use the torch backend for cpu tensors!')
        
        C = A.new(A.size())
        m, n = B.nelement() // 2, A.nelement() // B.nelement()
        lda = m
        ldc = m
        incx = 1
        handle = torch.cuda.current_blas_handle()
        stream = torch.cuda.current_stream()._as_parameter_
        cublas.cublasSetStream(handle, stream)
        cublas.cublasCdgmm(handle, 'l', m, n, A.data_ptr(), lda, B.data_ptr(), incx, C.data_ptr(), ldc)
        return C
    
    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        conjA = A.clone()
        conjB = B.clone()
        conjA[...,1] = -A[...,1]
        conjB[...,1] = -B[...,1]
        #conjA[:,:,:,:,1] = -A[:,:,:,:,1]
        #conjB[:,:,1] = -B[:,:,1]
        m, n = conjB.nelement() // 2, conjA.nelement() // conjB.nelement()
        # n is the B*C
        # m is the M*N
        gradA = conjA.new(conjA.size()) # (n,m), col-major
        gradC = grad_output.contiguous() # (n,m), col-major
        # grad_A = grad_C * conj(B)
        lda = m
        ldc = m
        incx = 1
        handle = torch.cuda.current_blas_handle()
        stream = torch.cuda.current_stream()._as_parameter_
        cublas.cublasSetStream(handle, stream)
        cublas.cublasCdgmm(handle, 'l', m, n, gradC.data_ptr(), lda, conjB.data_ptr(), incx, gradA.data_ptr(), ldc)
        
        # grad_B = sum_n grad_C * conj(A)
        # view grad_C and conjA as one vector of size n*m
        gradB_ = gradC.new(gradC.size()) # mul(gradC,conjA) # (B,C,M,N,2)
        lda = m*n
        ldc = m*n
        incx = 1
        cublas.cublasSetStream(handle, stream)
        cublas.cublasCdgmm(handle, 'l', m*n, 1, gradC.data_ptr(), lda, conjA.data_ptr(), incx, gradB_.data_ptr(), ldc)
        gradB = torch.sum(torch.sum(gradB_,0),0) # (m)
        
        return gradA, gradB


cdgmm = cdgmmMul.apply



class cdgmmMulcu(Function):
    @staticmethod
    def forward(ctx, A, B):
        # assume A and B has the same size , with last dim = 2
        A, B = A.contiguous(), B.contiguous()
        ctx.save_for_backward(A, B)
                        
        if not iscomplex(A) or not iscomplex(B):
            raise TypeError('The input, filter and output should be complex')

        if A.nelement() != B.nelement():
            raise TypeError('The input and filter should have same size')

        if type(A) is not type(B):
            raise RuntimeError('A and B should be same type!')

        if not A.is_cuda:
            raise RuntimeError('Use the torch backend for cpu tensors!')

        C = A.new(A.size())
        m, n = B.nelement() // 2, A.nelement() // B.nelement()
        lda = m
        ldc = m
        incx = 1
        handle = torch.cuda.current_blas_handle()
        stream = torch.cuda.current_stream()._as_parameter_
        cublas.cublasSetStream(handle, stream)
        cublas.cublasCdgmm(handle, 'l', m, n, A.data_ptr(), lda, B.data_ptr(), incx, C.data_ptr(), ldc)
        return C
    
    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        conjA = A.clone()
        conjB = B.clone()
        conjA[...,1] = -A[...,1]
        conjB[...,1] = -B[...,1]
        m, n = conjB.nelement() // 2, conjA.nelement() // conjB.nelement()
        # n is the B*C
        # m is the M*N
        gradA = conjA.new(conjA.size()) # (n,m), col-major
        #gradB = conjB.new(conjB.size()) # (m)
        gradC = grad_output.contiguous() # (n,m), col-major
        # grad_A = grad_C * conj(B)
        lda = m
        ldc = m
        incx = 1
        handle = torch.cuda.current_blas_handle()
        stream = torch.cuda.current_stream()._as_parameter_
        cublas.cublasSetStream(handle, stream)
        cublas.cublasCdgmm(handle, 'l', m, n, gradC.data_ptr(), lda, conjB.data_ptr(), incx, gradA.data_ptr(), ldc)
        
        # grad_B = sum_n grad_C * conj(A)
        # view grad_C and conjA as one vector of size n*m
        gradB = gradC.new(gradC.size()) # mul(gradC,conjA) # (...,2)
        lda = m*n
        ldc = m*n
        incx = 1
        #handle = torch.cuda.current_blas_handle()
        #stream = torch.cuda.current_stream()._as_parameter_
        cublas.cublasSetStream(handle, stream)
        cublas.cublasCdgmm(handle, 'l', m*n, 1, gradC.data_ptr(), lda, conjA.data_ptr(), incx, gradB.data_ptr(), ldc)

       # gradB_ = mul(gradC,conjA) # (B,C,M,N,2)
        #gradB = torch.sum(torch.sum(gradB_,0),0) # 
        
        return gradA, gradB
    
mulcu = cdgmmMulcu.apply

