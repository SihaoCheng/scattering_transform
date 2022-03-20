import os
import os.path as path
import time
import torch
from torch.autograd import Variable, Function

def is_long_tensor(z):
    return (isinstance(z, torch.LongTensor) or isinstance(z, torch.cuda.LongTensor))

def count_nans(z):
    print('\nNumber of NaNs:', ((z != z).sum().detach().cpu().numpy()), 'out of', z.size())
    raise SystemExit

class NanError(Exception):
    pass

class HookDetectNan(object):
    def __init__(self, message, *tensors):
        super(HookDetectNan, self).__init__()
        self.message = message
        self.tensors = tensors

    def __call__(self, grad):
        if (grad != grad).any():
            mask = grad != grad
            nan_source = '\n\n'.join([str(tensor[mask]) for tensor in self.tensors])
            print(nan_source)
            raise NanError("NaN detected in gradient: " + self.message)
            # print("NaN detected in gradient at time {}: {}".format(time.time(), self.message))
            # print((grad != grad).nonzero())


class HookPrintName(object):
    def __init__(self, message):
        super(HookPrintName, self).__init__()
        self.message = message

    def __call__(self, grad):
        print(self.message)


class MaskedFillZero(Function):
    @staticmethod
    def forward(ctx, input, mask):
        output = input.clone()
        output.masked_fill_(mask, 0)
        ctx.save_for_backward(mask)
        return output

    def backward(ctx, grad):
        mask, = ctx.saved_variables
        grad.masked_fill(mask, 0)
        return grad, None


masked_fill_zero = MaskedFillZero.apply
