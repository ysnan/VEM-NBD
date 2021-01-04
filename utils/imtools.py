''' kernel preparation and forward models '''
import numpy as np
from PIL import Image
from scipy.ndimage import filters
import torch
import torch.nn.functional as F
'''torch2np'''
def torch2np(x_tensor):
    if isinstance(x_tensor, np.ndarray):
        return x_tensor
    elif x_tensor.is_cuda == False:
        x = x_tensor.numpy()
        return x
    else:
        x = x_tensor.detach().cpu().numpy()
        return x

def np2torch(x, cuda=False):
    if isinstance(x, torch.Tensor):
        return x
    else:
        x = torch.from_numpy(x.copy())
        x = x.type(torch.float32)
        if cuda == True:
            x = x.cuda()
        return x



def for_fft(ker, shape):
    ker_mat = np.zeros(shape, dtype=np.float32)
    ker_shape = np.asarray(np.shape(ker))
    circ = np.ndarray.astype(-np.floor((ker_shape) / 2), dtype=np.int)
    ker_mat[:ker_shape[0], :ker_shape[1]] = ker
    ker_mat = np.roll(ker_mat, circ, axis=(0, 1))
    return ker_mat


'''convolution operator'''
def cconv_torch(x,ker):
    with torch.no_grad():
        x_h, x_v = x.size()
        conv_ker = np.flip(np.flip(ker, 0), 1)
        ker = torch.FloatTensor(conv_ker.copy()).cuda()
        k_h, k_v = ker.size()
        k2_h = k_h // 2
        k2_v = k_v // 2
        x = torch.cat((x[-k2_h:,:], x, x[0:k2_h,:]), dim = 0).cuda()
        x = torch.cat((x[:,-k2_v:], x, x[:,0:k2_v]), dim = 1).cuda()
        x = x.unsqueeze(0).cuda()
        x = x.unsqueeze(1).cuda()
        ker = ker.unsqueeze(0).cuda()
        ker = ker.unsqueeze(1).cuda()
        y1 = F.conv2d(x, ker).cuda()
        y1 = torch.squeeze(y1)
        y = y1[-x_h:, -x_v:]
    return y

def cconv_np(data, ker, mode='wrap'):
    # notice it might give false result when x is not the type.
    # Pay Attention, data and kernel is not interchangeable!!!
    if mode =='wrap':
        return filters.convolve(data, ker, mode='wrap')
    elif mode == 'valid':
        return fftconvolve(data, ker, mode='valid')

def imshow(x_in,str,dir = 'tmp/'):
    x = torch2np(x_in)
    x = np.squeeze(x)
    if len(x.shape) == 2:
        x[x > 1] = 1
        x[x < 0] = 0
        x_int = np.uint8(np.around(x * 255))
        Image.fromarray(x_int, 'L').save(dir + str + '.png')
    elif len(x.shape) == 3:
        x[x > 1] = 1
        x[x < 0] = 0
        x_int = np.uint8(np.around(x * 255))
        Image.fromarray(x_int,'RGB').save(dir + str + '.png')



from scipy.signal import fftconvolve
def pad_for_kernel(img, kernel, mode):
    hy = (kernel.shape[0] - 1) // 2
    hx = (kernel.shape[0] - 1) - hy
    wy = (kernel.shape[1] - 1) // 2
    wx = (kernel.shape[1] - 1) - wy
    # p = [(d - 1) // 2 for d in kernel.shape]
    padding = [[hx, hy], [wx, wy]]
    return np.pad(img, padding, mode)
def edgetaper(img, kernel, n_tapers=3):
    '''tap edges for immitation of circulant boundary. '''
    alpha = edgetaper_alpha(kernel, img.shape)
    _kernel = kernel
    for i in range(n_tapers):
        blurred = fftconvolve(pad_for_kernel(img, _kernel, 'wrap'), kernel, mode='valid')
        img = alpha * img + (1 - alpha) * blurred
    return img
def edgetaper_alpha(kernel,img_shape):
    v = []
    for i in range(2):
        z = np.fft.fft(np.sum(kernel,1-i),img_shape[i]-1)
        z = np.real(np.fft.ifft(np.square(np.abs(z)))).astype(np.float32)
        z = np.concatenate([z,z[0:1]],0)
        v.append(1 - z/np.max(z))
    return np.outer(*v)


