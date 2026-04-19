import numpy as np


def conv2d_forward_single_channel(x: np.ndarray, kernel: np.ndarray, bias: float = 0.0, stride: int = 1, padding: int = 0):
    """
    Naive 2D convolution forward for single-channel input and single kernel.

    Parameters
    ----------
    x : np.ndarray of shape (H, W)
    kernel : np.ndarray of shape (kH, kW)
    bias : float
    stride : int
    padding : int
    """
    assert x.ndim == 2
    assert kernel.ndim == 2

    if padding > 0:
        x = np.pad(x, ((padding, padding), (padding, padding)), mode="constant")

    H, W = x.shape
    kH, kW = kernel.shape

    out_h = (H - kH) // stride + 1
    out_w = (W - kW) // stride + 1

    out = np.zeros((out_h, out_w), dtype=np.float64)

    for i in range(out_h):
        for j in range(out_w):
            hs = i * stride
            ws = j * stride
            window = x[hs:hs + kH, ws:ws + kW]
            out[i, j] = np.sum(window * kernel) + bias

    return out


def conv2d_forward_multi_in(x: np.ndarray, weight: np.ndarray, bias: np.ndarray = None, stride: int = 1, padding: int = 0):
    """
    Multi-input multi-output convolution forward.

    Parameters
    ----------
    x : np.ndarray of shape (C_in, H, W)
    weight : np.ndarray of shape (C_out, C_in, kH, kW)
    bias : np.ndarray of shape (C_out,)
    """
    assert x.ndim == 3
    assert weight.ndim == 4

    c_out, c_in, kH, kW = weight.shape
    assert x.shape[0] == c_in

    if padding > 0:
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode="constant")

    _, H, W = x.shape
    out_h = (H - kH) // stride + 1
    out_w = (W - kW) // stride + 1

    out = np.zeros((c_out, out_h, out_w), dtype=np.float64)

    if bias is None:
        bias = np.zeros(c_out, dtype=np.float64)

    for oc in range(c_out):
        for i in range(out_h):
            for j in range(out_w):
                hs = i * stride
                ws = j * stride
                val = 0.0
                for ic in range(c_in):
                    window = x[ic, hs:hs + kH, ws:ws + kW]
                    val += np.sum(window * weight[oc, ic])
                out[oc, i, j] = val + bias[oc]

    return out


def maxpool2d_forward(x: np.ndarray, kernel_size: int = 2, stride: int = 2):
    """
    Max pooling forward for input of shape (C, H, W).
    """
    assert x.ndim == 3

    C, H, W = x.shape
    out_h = (H - kernel_size) // stride + 1
    out_w = (W - kernel_size) // stride + 1

    out = np.zeros((C, out_h, out_w), dtype=np.float64)

    for c in range(C):
        for i in range(out_h):
            for j in range(out_w):
                hs = i * stride
                ws = j * stride
                window = x[c, hs:hs + kernel_size, ws:ws + kernel_size]
                out[c, i, j] = np.max(window)

    return out