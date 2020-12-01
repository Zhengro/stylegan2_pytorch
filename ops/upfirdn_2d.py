"""Custom PyTorch ops for efficient resampling of 2D images."""

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
def upfirdn_2d(x, k, upx=1, upy=1, downx=1, downy=1, padx0=0, padx1=0, pady0=0, pady1=0, impl='ref'):
    """
    :param x: Input tensor of the shape `[majorDim, inH, inW, minorDim]`.
    :param k: 2D FIR filter of the shape `[firH, firW]`.
    :param upx: Integer upsampling factor along the X-axis (default: 1).
    :param upy: Integer upsampling factor along the Y-axis (default: 1).
    :param downx: Integer downsampling factor along the X-axis (default: 1).
    :param downy: Integer downsampling factor along the Y-axis (default: 1).
    :param padx0: Number of pixels to pad on the left side (default: 0).
    :param padx1: Number of pixels to pad on the right side (default: 0).
    :param pady0: Number of pixels to pad on the top side (default: 0).
    :param pady1: Number of pixels to pad on the bottom side (default: 0).
    :param impl: Name of the implementation to use. Can be `"ref"` (default) or `"cuda"`.
    :return: Tensor of the shape `[majorDim, outH, outW, minorDim]`, and same datatype as `x`.
    """

    impl_dict = {
        'ref':  _upfirdn_2d_ref,
        'cuda': _upfirdn_2d_cuda,  # TODO: implement this and use it as default
    }
    return impl_dict[impl](x=x, k=k, upx=upx, upy=upy, downx=downx, downy=downy, padx0=padx0, padx1=padx1, pady0=pady0, pady1=pady1)


# ----------------------------------------------------------------------------
def _upfirdn_2d_ref(x, k, upx, upy, downx, downy, padx0, padx1, pady0, pady1):
    """Slow reference implementation of `upfirdn_2d()` using standard PyTorch ops."""

    _, inH, inW, minorDim = x.shape
    kernelH, kernelW = k.shape
    k = torch.as_tensor(k)

    # Upsample (insert zeros).
    x = x.reshape(-1, inH, 1, inW, 1, minorDim)
    x = F.pad(x, (0, 0, 0, upx - 1, 0, 0, 0, upy - 1))
    x = x.reshape(-1, inH * upy, inW * upx, minorDim)

    # Pad (crop if negative).
    x = F.pad(x, (0, 0, max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)))
    x = x[:, max(-pady0, 0):x.shape[1] - max(-pady1, 0), max(-padx0, 0):x.shape[2] - max(-padx1, 0), :]

    # Convolve with filter.
    x = x.permute(0, 3, 1, 2)
    x = x.reshape(-1, 1, inH * upy + pady0 + pady1, inW * upx + padx0 + padx1)
    w = torch.flip(k, [0, 1]).view(1, 1, kernelH, kernelW).to(x.device)
    x = F.conv2d(x, w)
    x = x.reshape(-1, minorDim, inH * upy + pady0 + pady1 - kernelH + 1, inW * upx + padx0 + padx1 - kernelW + 1)
    x = x.permute(0, 2, 3, 1)

    # Downsample (throw away pixels).
    return x[:, ::downy, ::downx, :]


# ----------------------------------------------------------------------------
def _upfirdn_2d_cuda(x, k, upx, upy, downx, downy, padx0, padx1, pady0, pady1):
    """Fast CUDA implementation of `upfirdn_2d()` using custom ops."""

    pass


# ----------------------------------------------------------------------------
def upsample_2d(x, k=None, factor=2, gain=1, impl='ref'):
    """
    :param x: Input tensor of the shape `[N, C, H, W]`.
    :param k: FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
              The default is `[1] * factor`, which corresponds to nearest-neighbor upsampling.
    :param factor: Integer upsampling factor (default: 2).
    :param gain: Scaling factor for signal magnitude (default: 1.0).
    :param impl: Name of the implementation to use. Can be `"ref"` (default) or `"cuda"`.
    :return: Tensor of the shape `[N, C, H * factor, W * factor]`, and same datatype as `x`
    """

    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor ** 2))
    p = k.shape[0] - factor
    return _simple_upfirdn_2d(x, k, up=factor, pad0=(p + 1) // 2 + factor - 1, pad1=p // 2, impl=impl)


# ----------------------------------------------------------------------------
def downsample_2d(x, k=None, factor=2, gain=1, impl='ref'):
    """
    :param x: Input tensor of the shape `[N, C, H, W]`.
    :param k: FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
              The default is `[1] * factor`, which corresponds to average pooling.
    :param factor: Integer downsampling factor (default: 2).
    :param gain: Scaling factor for signal magnitude (default: 1.0).
    :param impl: Name of the implementation to use. Can be `"ref"` (default) or `"cuda"`.
    :return: Tensor of the shape `[N, C, H // factor, W // factor]`, and same datatype as `x`.
    """

    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = k.shape[0] - factor
    return _simple_upfirdn_2d(x, k, down=factor, pad0=(p + 1) // 2, pad1=p // 2, impl=impl)


# ----------------------------------------------------------------------------
def upsample_conv_2d(x, w, k=None, factor=2, gain=1, impl='ref'):
    """
    :param x: Input tensor of the shape `[N, C, H, W]`.
    :param w: Weight tensor of the shape `[outChannels, inChannels, filterH, filterW]`.
    :param k: FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
              The default is `[1] * factor`, which corresponds to nearest-neighbor upsampling.
    :param factor: Integer upsampling factor (default: 2).
    :param gain: Scaling factor for signal magnitude (default: 1.0).
    :param impl: Name of the implementation to use. Can be `"ref"` (default) or `"cuda"`.
    :return: Tensor of the shape `[N, C, H * factor, W * factor]`, and same datatype as `x`.
    """

    _, inC, convH, convW = w.shape

    # Setup filter kernel.
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor ** 2))
    p = (k.shape[0] - factor) - (convW - 1)

    # Determine data dimensions.
    num_groups = x.shape[1] // inC  # batch size

    # Transpose weights.
    w = w.view(num_groups, -1, inC, convH, convW)
    w = torch.flip(w, [3, 4]).permute(0, 2, 1, 3, 4)
    w = w.reshape(num_groups * inC, -1, convH, convW)

    # Execute.
    x = F.conv_transpose2d(x, w, stride=factor, groups=num_groups)
    x = x.view(num_groups, -1, x.shape[2], x.shape[3])
    return _simple_upfirdn_2d(x, k, pad0=(p + 1) // 2 + factor - 1, pad1=p // 2 + 1, impl=impl)


# ----------------------------------------------------------------------------
def conv_downsample_2d(x, w, k=None, factor=2, gain=1, impl='ref'):
    """
    :param x: Input tensor of the shape `[N, C, H, W]`.
    :param w: Weight tensor of the shape `[outChannels, inChannels, filterH, filterW]`.
    :param k: FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
              The default is `[1] * factor`, which corresponds to average pooling.
    :param factor: Integer downsampling factor (default: 2).
    :param gain: Scaling factor for signal magnitude (default: 1.0).
    :param impl: Name of the implementation to use. Can be `"ref"` (default) or `"cuda"`.
    :return: Tensor of the shape `[N, C, H // factor, W // factor]`, and same datatype as `x`.
    """

    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = (k.shape[0] - factor) + (w.shape[-1] - 1)

    num_groups = x.shape[1] // w.shape[1]  # batch size

    x = x.view(num_groups, -1, x.shape[2], x.shape[3])
    x = _simple_upfirdn_2d(x, k, pad0=(p + 1) // 2, pad1=p // 2, impl=impl)
    x = x.view(1, -1, x.shape[2], x.shape[3])
    x = F.conv2d(x, w, stride=factor, groups=num_groups)
    return x.view(num_groups, -1, x.shape[2], x.shape[3])


# ----------------------------------------------------------------------------
def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k


def _simple_upfirdn_2d(x, k, up=1, down=1, pad0=0, pad1=0, impl='ref'):
    y = x
    y = y.reshape(-1, y.shape[2], y.shape[3], 1)
    y = upfirdn_2d(y, k, upx=up, upy=up, downx=down, downy=down, padx0=pad0, padx1=pad1, pady0=pad0, pady1=pad1, impl=impl)
    y = y.reshape(-1, x.shape[1], y.shape[1], y.shape[2])
    return y
