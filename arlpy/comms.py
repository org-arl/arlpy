"""Communications toolbox."""

import numpy as _np
import scipy.signal as _sp

# set up population count table for fast BER computation
_MAX_M = 64
_popcount = _np.empty(_MAX_M, dtype=_np.int)
for _i in range(_MAX_M):
    _popcount[_i] = bin(_i).count('1')

def random_data(size, m=2):
    """Generate random integers in the range [0, m-1]."""
    return _np.random.randint(0, m, size)

def gray_code(m):
    """Generate a Gray code map of size m."""
    x = range(m)
    x = map(lambda a: a ^ (a >> 1), x)
    return _np.asarray(x)

def invert_map(x):
    """Generate an inverse map."""
    y = _np.empty_like(x)
    y[x] = _np.arange(len(x))
    return y

def bi2sym(x, m):
    """Convert bits to symbols."""
    n = int(_np.log2(m))
    if 2**n != m:
        raise ValueError('m must be a power of 2')
    x = _np.asarray(x, dtype=_np.int)
    if _np.any(x < 0) or _np.any(x > 1):
        raise ValueError('Invalid data bits')
    y = _np.zeros(len(x)/n, dtype=_np.int)
    a = 0
    for i in range(len(x)):
        a = (a << 1) | x[i]
        if i % n == n-1:
            y[i/n] = a
            a = 0
    return y

def sym2bi(x, m):
    """Convert symbols to bits."""
    n = int(_np.log2(m))
    if 2**n != m:
        raise ValueError('m must be a power of 2')
    x = _np.asarray(x, dtype=_np.int)
    if _np.any(x < 0) or _np.any(x >= m):
        raise ValueError('Invalid data for specified m')
    y = _np.zeros(n*len(x), dtype=_np.int)
    for i in range(len(x)):
        y[n*i:n*(i+1)] = [1 if d=='1' else 0 for d in bin(x[i]|(1<<n))[3:]]
    return y

def pam(m=2, gray=True):
    """Generate a PAM constellation with m signal points."""
    x = _np.arange(m, dtype=_np.float)
    x -= _np.mean(x)
    x /= _np.std(x)
    if gray:
        x = x[invert_map(gray_code(m))]
    return x

def psk(m=2, phase0=None, gray=True):
    """Generate a PSK constellation with m signal points."""
    if phase0 is None:
        phase0 = _np.pi/4 if m == 4 else 0
    x = _np.round(_np.exp(1j*(2*_np.pi/m*_np.arange(m)+phase0)), decimals=8)
    if gray:
        x = x[invert_map(gray_code(m))]
    return x

def qam(m=16, gray=True):
    """Generate a QAM constellation with m signal points."""
    n = int(_np.sqrt(m))
    if n*n != m:
        raise ValueError('m must be an integer squared')
    x = _np.empty((n, n), dtype=_np.complex)
    for r in range(n):
        for i in range(n):
            x[r,i] = r + 1j*i
    x -= _np.mean(x)
    x /= _np.std(x)
    x = _np.ravel(x)
    if gray:
        ndx = _np.reshape(gray_code(m), (n,n))
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
    m = len(const)
    data = data.astype(int)
    if _np.any(data > m-1) or _np.any(data < 0):
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

def ber(x, y, m=2):
    """Measure bit error rate between symbols in x and y."""
    x = _np.asarray(x, dtype=_np.int)
    y = _np.asarray(y, dtype=_np.int)
    if _np.any(x >= m) or _np.any(y >= m) or _np.any(x < 0) or _np.any(y < 0):
        raise ValueError('Invalid data for specified m')
    if m == 2:
        return ser(x, y)
    if m > _MAX_M:
        raise ValueError('m > %d not supported' % (_MAX_M))
    n = _np.product(_np.shape(x))*_np.log2(m)
    e = x^y
    e = e[_np.nonzero(e)]
    e = _np.sum(_popcount[e])
    return float(e)/n
