"""Signal processing toolbox."""

import operator as _op
import numpy as _np
import scipy.signal as _sig

def time(n, fs):
    """Generate a time vector for time series with n data points."""
    return _np.arange(n, dtype=_np.float)/fs

def envelope(x):
    """Generate a Hilbert envelope of the real signal x."""
    return _np.abs(_sig.hilbert(x, axis=0))

def mseq(spec, n=None):
    """Generate m-sequence with given specifier and length n."""
    if isinstance(spec, int):
        known_specs = {  # known m-sequences are specified as base 1 taps
             2: [1,2],          3: [1,3],          4: [1,4],          5: [2,5],
             6: [1,6],          7: [1,7],          8: [1,2,7,8],      9: [4,9],
            10: [3,10],        11: [9,11],        12: [6,8,11,12],   13: [9,10,12,13],
            14: [4,8,13,14],   15: [14,15],       16: [4,13,15,16],  17: [14,17],
            18: [11,18],       19: [14,17,18,19], 20: [17,20],       21: [19,21],
            22: [21,22],       23: [18,23],       24: [17,22,23,24], 25: [22,25],
            26: [20,24,25,26], 27: [22,25,26,27], 28: [25,28],       29: [27,29],
            30: [7,28,29,30]
        }
        spec = map(lambda x: x-1, known_specs[spec])  # convert to base 0 taps
    spec.sort(reverse=True)
    m = spec[0]+1
    if n is None:
        n = 2**m-1
    reg = _np.ones(m, dtype=_np.uint8)
    out = _np.zeros(n)
    for j in range(n):
        b = reduce(_op.xor, reg[spec], 0)
        reg = _np.roll(reg, 1)
        out[j] = float(2*reg[0]-1)
        reg[0] = b
    return out

def freqz(b, a=1, fs=2.0, worN=None, whole=False):
    """Plot frequency response of filter."""
    import matplotlib.pyplot as plt
    w, h = _sig.freqz(b, a, worN, whole)
    f = w*fs/(2*_np.pi)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.plot(f, 20*_np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [Hz]')
    ax2 = ax1.twinx()
    angles = _np.unwrap(_np.angle(h))
    plt.plot(f, angles, 'g')
    plt.ylabel('Angle (radians)', color='g')
    plt.grid()
    plt.axis('tight')
    plt.show()
    return w, h

def bb2pb(x, fd, fc, fs=None):
    """Convert baseband signal x sampled at fd to passband with center frequency fc and sampling rate fs."""
    if fs is None or fs == fd:
        y = _np.array(x, dtype=_np.complex)
        fs = fd
    else:
        y = _sig.resample(_np.asarray(x, dtype=_np.complex), _np.round(float(fs)/fd*len(x)))
    y *= _np.sqrt(2)*_np.exp(-2j*_np.pi*fc*time(len(y),fs))
    return _np.real(y)

def pb2bb(x, fs, fc, fd=None):
    """Convert passband signal x sampled at fs to baseband with center frequency fc and sampling rate fd."""
    y = x * _np.sqrt(2)*_np.exp(2j*_np.pi*fc*time(len(x),fs))
    flen = int(_np.ceil(float(fs)/fc))
    hb = _np.ones(flen)/flen
    y = _sig.filtfilt(hb, 1, y)
    if fd is not None and fd != fs:
        y = _sig.resample(y, _np.round(float(fd)/fs*len(y)))
    return y

def matchedfilter(x, s):
    """Matched filter recevied signal x using reference signal s."""
    if _np.ndim(x) == 1 and _np.ndim(s) == 1:
        y = _np.correlate(x, s, 'full')
        return y[len(s)-1:len(s)-1+len(x)]
    if _np.ndim(x) == 1 and _np.ndim(s) == 2:
        [m,n] = _np.shape(s)
        z = _np.zeros([len(x), n])
        for j in range(n):
            y = _np.correlate(x, s[:,j], 'full')
            z[:,j] = y[m-1:m-1+len(x)]
        return z
    if _np.ndim(x) == 2 and _np.ndim(s) == 1:
        [m,n] = _np.shape(x)
        z = _np.zeros([m, n])
        for j in range(n):
            y = _np.correlate(x[:,j], s, 'full')
            z[:,j] = y[len(s)-1:len(s)-1+m]
        return z
    raise ValueError('Either of x or s must be a vector')
