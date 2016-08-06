"""Communications toolbox."""

import numpy as _np
import scipy.signal as _sp

# set up population count table for fast BER computation
_maxM = 64
_popcount = _np.empty(_maxM, dtype=_np.int)
for _i in range(_maxM):
    _popcount[_i] = bin(_i).count('1')

def random_data(size, M=2):
    """Generate random integers in the range [0, M-1]."""
    return _np.random.randint(0, M, size)

def gray_code(M):
    """Generate a Gray code map of size M."""
    x = range(M)
    x = map(lambda a: a ^ (a >> 1), x)
    return _np.asarray(x)

def invert_map(x):
    """Generate an inverse map."""
    y = _np.empty_like(x)
    y[x] = _np.arange(len(x))
    return y

def pam(M=2, gray=True):
    """Generate a PAM constellation with M signal points."""
    x = _np.arange(M, dtype=_np.float)
    x -= _np.mean(x)
    x /= _np.std(x)
    if gray:
        x = x[invert_map(gray_code(M))]
    return x

def psk(M=2, phase0=None, gray=True):
    """Generate a PSK constellation with M signal points."""
    if phase0 is None:
        phase0 = _np.pi/4 if M == 4 else 0
    x = _np.round(_np.exp(1j*(2*_np.pi/M*_np.arange(M)+phase0)), decimals=8)
    if gray:
        x = x[invert_map(gray_code(M))]
    return x

def qam(M=16, gray=True):
    """Generate a QAM constellation with M signal points."""
    n = int(_np.sqrt(M))
    if n*n != M:
        raise ValueError('M must be an integer squared')
    x = _np.empty((n, n), dtype=_np.complex)
    for r in range(n):
        for i in range(n):
            x[r,i] = r + 1j*i
    x -= _np.mean(x)
    x /= _np.std(x)
    x = _np.ravel(x)
    if gray:
        ndx = _np.reshape(gray_code(M), (n,n))
        for i in range(1, n, 2):
            ndx[i] = _np.flipud(ndx[i])
        ndx = invert_map(_np.ravel(ndx))
        x = x[ndx]
    return x

def iqplot(data, spec='.', labels=None):
    """Plot signal points."""
    import matplotlib.pyplot as plt
    data = _np.asarray(data)
    if labels is None:
        plt.plot(data.real, data.imag, spec)
    else:
        if labels == True:
            labels = range(len(data))
        for i in range(len(data)):
            plt.text(data[i].real, data[i].imag, str(labels[i]))
    plt.axis([-2, 2, -2, 2])
    plt.grid()
    plt.show()

def modulate(data, const):
    """Modulate data into signal points for the specified constellation."""
    M = len(const)
    data = data.astype(int)
    if _np.any(data > M-1) or _np.any(data < 0):
        raise ValueError('Invalid data for specified constellation')
    const = _np.asarray(const)
    f = _np.vectorize(lambda x: const[x], otypes=[const.dtype])
    return f(data)

def demodulate(x, const):
    """Demodulate complex signal based on the specified constellation."""
    f = _np.vectorize(lambda y: _np.argmin(_np.abs(y-const)), otypes=[_np.int])
    return f(x)

def awgn(x, snr, measured=False):
    """Add Gaussian noise to signal."""
    signal = _np.std(x) if measured else 1.0
    noise = signal * _np.power(10, -snr/20.0)
    if x.dtype == _np.complex:
        noise /= _np.sqrt(2)
        y = x + _np.random.normal(0, noise, _np.shape(x)) + 1j*_np.random.normal(0, noise, _np.shape(x))
    else:
        y = x + _np.random.normal(0, noise, _np.shape(x))
    return y

def ser(x, y):
    """Measure symbol error rate between symbols in x and y."""
    x = _np.asarray(x, dtype=_np.int)
    y = _np.asarray(y, dtype=_np.int)
    n = _np.product(_np.shape(x))
    e = _np.count_nonzero(x^y)
    return float(e)/n

def ber(x, y, M=2):
    """Measure bit error rate between symbols in x and y."""
    x = _np.asarray(x, dtype=_np.int)
    y = _np.asarray(y, dtype=_np.int)
    if _np.any(x >= M) or _np.any(y >= M) or _np.any(x < 0) or _np.any(y < 0):
        raise ValueError('Invalid data for specified M')
    if M == 2:
        return ser(x, y)
    if M > _maxM:
        raise ValueError('M > %d not supported' % (_maxM))
    n = _np.product(_np.shape(x))*_np.log2(M)
    e = x^y
    e = e[_np.nonzero(e)]
    e = _np.sum(_popcount[e])
    return float(e)/n
