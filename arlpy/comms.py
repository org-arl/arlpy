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
    nsym = len(x)/n
    x = _np.reshape(x, (nsym, n))
    y = _np.zeros(nsym, dtype=_np.int)
    for i in range(n):
        y <<= 1
        y |= x[:, i]
    return y

def sym2bi(x, m):
    """Convert symbols to bits."""
    n = int(_np.log2(m))
    if 2**n != m:
        raise ValueError('m must be a power of 2')
    x = _np.asarray(x, dtype=_np.int)
    if _np.any(x < 0) or _np.any(x >= m):
        raise ValueError('Invalid data for specified m')
    y = _np.zeros((len(x), n), dtype=_np.int)
    for i in range(n):
        y[:, n-i-1] = (x >> i) & 1
    return _np.ravel(y)

def ook():
    """Generate an OOK constellation."""
    return _np.array([0, _np.sqrt(2)], dtype=_np.float)

def pam(m=2, gray=True, centered=True):
    """Generate a PAM constellation with m signal points."""
    x = _np.arange(m, dtype=_np.float)
    if centered:
        x -= _np.mean(x)
    x /= _np.sqrt(_np.mean(x**2))
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

def fsk(m=2, n=None):
    """Generate a m-FSK constellation with n baseband samples per symbol."""
    if n is None:
        n = m
    if n < m:
        raise ValueError('n must be >= m')
    f = _np.linspace(-1.0, 1.0, m) * (0.5-0.5/m)
    x = _np.empty((m, n), dtype=_np.complex)
    for i in range(m):
        x[i] = _np.exp(-2j*_np.pi*f[i]*_np.arange(n))
    return x

def msk():
    """Generate a MSK constellation with 4 baseband samples per 2-bit symbol."""
    return _np.array([[1,  1j, -1, -1j],
                      [1,  1j, -1,  1j],
                      [1, -1j, -1, -1j],
                      [1, -1j, -1,  1j]], dtype=_np.complex)

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
    data = _np.asarray(data, dtype=_np.int)
    const = _np.asarray(const)
    return _np.ravel(const[data])

def demodulate(x, const, metric=None, decision=lambda a: _np.argmin(a, axis=1)):
    """Demodulate complex signal based on the specified constellation."""
    if metric is None:
        if const.ndim == 2:
            # multi-dimensional constellation => matched filter
            m, n = const.shape
            metric = lambda a, b: -_np.abs(_np.sum(_np.expand_dims(_np.reshape(a,(len(x)/n, n)), axis=2) * b.conj().T, axis=1))
        elif _np.all(_np.abs(const.imag) < 1e-6) and _np.all(const.real >= 0):
            # all real constellation => incoherent distance
            metric = lambda a, b: _np.abs(_np.abs(a)-b)
        else:
            # general PSK/QAM constellation => Euclidean distance
            metric = lambda a, b: _np.abs(a-b)
    y = metric(_np.expand_dims(x, axis=1), const)
    return y if decision is None else decision(y)

def diff_encode(x):
    """Encode phase differential baseband signal."""
    x = _np.asarray(x)
    y = _np.insert(x, 0, 1)
    for j in range(2,len(y)):
        y[j] *= y[j-1]
    return y

def diff_decode(x):
    """Decode phase differential baseband signal."""
    x = _np.asarray(x)
    y = _np.array(x)
    y[1:] *= x[:-1].conj()
    return y[1:]

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
