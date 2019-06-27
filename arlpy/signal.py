##############################################################################
#
# Copyright (c) 2016, Mandar Chitre
#
# This file is part of arlpy which is released under Simplified BSD License.
# See file LICENSE or go to http://www.opensource.org/licenses/BSD-3-Clause
# for full license details.
#
##############################################################################

"""Signal processing toolbox."""

import functools
import operator as _op
import numpy as _np
import scipy.signal as _sig
import arlpy.utils as _utils

def time(n, fs):
    """Generate a time vector for time series.

    :param n: time series, or number of samples
    :param fs: sampling rate in Hz
    :returns: time vector starting at time 0

    >>> import arlpy
    >>> t = arlpy.signal.time(100000, fs=250000)
    >>> t
    array([  0.00000000e+00,   4.00000000e-06,   8.00000000e-06, ...,
         3.99988000e-01,   3.99992000e-01,   3.99996000e-01])
    >>> x = arlpy.signal.cw(fc=27000, duration=0.5, fs=250000)
    >>> t = arlpy.signal.time(x, fs=250000)
    >>> t
    array([  0.00000000e+00,   4.00000000e-06,   8.00000000e-06, ...,
         4.99988000e-01,   4.99992000e-01,   4.99996000e-01])
    """
    if hasattr(n, "__len__"):
        n = _np.asarray(n).shape[0]
    return _np.arange(n, dtype=_np.float)/fs

def cw(fc, duration, fs, window=None, complex_output=False):
    """Generate a sinusoidal pulse.

    :param fc: frequency of the pulse in Hz
    :param duration: duration of the pulse in s
    :param fs: sampling rate in Hz
    :param window: window function to use (``None`` means rectangular window)
    :param complex_output: True to return complex signal, False for a real signal

    For supported window functions, see documentation for :func:`scipy.signal.get_window`.

    >>> import arlpy
    >>> x1 = arlpy.signal.cw(fc=27000, duration=0.5, fs=250000)
    >>> x2 = arlpy.signal.cw(fc=27000, duration=0.5, fs=250000, window='hamming')
    >>> x3 = arlpy.signal.cw(fc=27000, duration=0.5, fs=250000, window=('kaiser', 4.0))
    """
    n = int(round(duration*fs))
    x = _np.exp(2j*_np.pi*fc*time(n, fs)) if complex_output else _np.sin(2*_np.pi*fc*time(n, fs))
    if window is not None:
        w = _sig.get_window(window, n, False)
        x *= w
    return x

def sweep(f1, f2, duration, fs, method='linear', window=None):
    """Generate frequency modulated sweep.

    :param f1: starting frequency in Hz
    :param f2: ending frequency in Hz
    :param duration: duration of the pulse in s
    :param fs: sampling rate in Hz
    :param method: type of sweep (``'linear'``, ``'quadratic'``, ``'logarithmic'``, ``'hyperbolic'``)
    :param window: window function to use (``None`` means rectangular window)

    For supported window functions, see documentation for :func:`scipy.signal.get_window`.

    >>> import arlpy
    >>> x1 = arlpy.signal.sweep(20000, 30000, duration=0.5, fs=250000)
    >>> x2 = arlpy.signal.sweep(20000, 30000, duration=0.5, fs=250000, window='hamming')
    >>> x2 = arlpy.signal.sweep(20000, 30000, duration=0.5, fs=250000, window=('kaiser', 4.0))
    """
    n = int(round(duration*fs))
    x = _sig.chirp(time(n, fs), f1, duration, f2, method)
    if window is not None:
        w = _sig.get_window(window, n, False)
        x *= w
    return x

def envelope(x, axis=-1):
    """Generate a Hilbert envelope of the real signal x.

    :param x: real passband signal
    :param axis: axis of the signal, if multiple signals specified
    """
    return _np.abs(_sig.hilbert(x, axis=axis))

def mseq(spec, n=None):
    """Generate m-sequence.

    m-sequences are sequences of :math:`\\pm 1` values with near-perfect discrete periodic
    auto-correlation properties. All non-zero lag periodic auto-correlations
    are -1. The zero-lag autocorrelation is :math:`2^m-1`, where m is the shift register
    length.

    This function currently supports shift register lengths between 2 and 30.

    :param spec: m-sequence specifier (shift register length or taps)
    :param n: length of sequence (``None`` means full length of :math:`2^m-1`)

    >>> import arlpy
    >>> x = arlpy.signal.mseq(7)
    >>> len(x)
    127
    """
    if isinstance(spec, int):
        if spec < 2 or spec > 30:
            raise ValueError('spec must be between 2 and 30')
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
        spec = list(map(lambda x: x-1, known_specs[spec]))  # convert to base 0 taps
    spec.sort(reverse=True)
    m = spec[0]+1
    if n is None:
        n = 2**m-1
    reg = _np.ones(m, dtype=_np.uint8)
    out = _np.zeros(n)
    for j in range(n):
        b = functools.reduce(_op.xor, reg[spec], 0)
        reg = _np.roll(reg, 1)
        out[j] = float(2*reg[0]-1)
        reg[0] = b
    return out

def gmseq(spec, theta=None):
    """Generate generalized m-sequence.

    Generalized m-sequences are related to m-sequences but have an additional parameter
    :math:`\\theta`. When :math:`\\theta = \\pi/2`, generalized m-sequences become normal m-sequences. When
    :math:`\\theta < \\pi/2`, generalized m-sequences contain a DC-component that leads to an exalted
    carrier after modulation.

    When theta is :math:`\\arctan(\\sqrt{n})` where :math:`n` is the length of the m-sequence, the m-sequence
    is considered to be period matched. Period matched m-sequences are complex sequences
    with perfect discrete periodic auto-correlation properties, i.e., all non-zero lag
    periodic auto-correlations are zero. The zero-lag autocorrelation is :math:`n = 2^m-1`, where
    m is the shift register length.

    This function currently supports shift register lengths between 2 and 30.

    :param spec: m-sequence specifier (shift register length or taps)
    :param theta: transmission angle (``None`` to use period-matched angle)

    >>> import arlpy
    >>> x = arlpy.signal.gmseq(7)
    >>> len(x)
    127
    """
    x = mseq(spec)
    if theta is None:
        theta = _np.arctan(_np.sqrt(len(x)))
    return _np.cos(theta) + 1j*_np.sin(theta)*x

def bb2pb(x, fd, fc, fs=None, axis=-1):
    """Convert baseband signal to passband.

    For communication applications, one may wish to use :func:`arlpy.comms.upconvert` instead,
    as that function supports pulse shaping.

    The convention used in that exp(2j*pi*fc*t) is a positive frequency carrier.

    :param x: complex baseband signal
    :param fd: sampling rate of baseband signal in Hz
    :param fc: carrier frequency in passband in Hz
    :param fs: sampling rate of passband signal in Hz (``None`` => same as `fd`)
    :param axis: axis of the signal, if multiple signals specified
    :returns: real passband signal, sampled at `fs`
    """
    if fs is None or fs == fd:
        y = _np.array(x, dtype=_np.complex)
        fs = fd
    else:
        y = _sig.resample_poly(_np.asarray(x, dtype=_np.complex), fs, fd, axis=axis)
    osc = _np.sqrt(2)*_np.exp(2j*_np.pi*fc*time(y,fs))
    y *= _utils.broadcastable_to(osc, y.shape, axis)
    return y.real

def pb2bb(x, fs, fc, fd=None, flen=127, cutoff=None, axis=-1):
    """Convert passband signal to baseband.

    The baseband conversion uses a low-pass filter after downconversion, with a
    default cutoff frequency of `0.6*fd`, if `fd` is specified, or `1.1*fc` if `fd`
    is not specified. Alternatively, the user may specify the cutoff frequency
    explicitly.

    For communication applications, one may wish to use :func:`arlpy.comms.downconvert` instead,
    as that function supports matched filtering with a pulse shape rather than a generic
    low-pass filter.

    The convention used in that exp(2j*pi*fc*t) is a positive frequency carrier.

    :param x: passband signal
    :param fs: sampling rate of passband signal in Hz
    :param fc: carrier frequency in passband in Hz
    :param fd: sampling rate of baseband signal in Hz (``None`` => same as `fs`)
    :param flen: number of taps in the low-pass FIR filter
    :param cutoff: cutoff frequency in Hz (``None`` means auto-select)
    :param axis: axis of the signal, if multiple signals specified
    :returns: complex baseband signal, sampled at `fd`
    """
    if cutoff is None:
        cutoff = 0.6*fd if fd is not None else 1.1*_np.abs(fc)
    osc = _np.sqrt(2)*_np.exp(-2j*_np.pi*fc*time(x.shape[axis],fs))
    y = x * _utils.broadcastable_to(osc, x.shape, axis)
    hb = _sig.firwin(flen, cutoff=cutoff, nyq=fs/2.0)
    y = _sig.filtfilt(hb, 1, y, axis=axis)
    if fd is not None and fd != fs:
        y = _sig.resample_poly(y, 2*fd, fs, axis=axis)
        y = _np.apply_along_axis(lambda a: a[::2], axis, y)
    return y

def mfilter(s, x, complex_output=False, axis=-1):
    """Matched filter recevied signal using a reference signal.

    :param s: reference signal
    :param x: recevied signal
    :param complex_output: True to return complex signal, False for absolute value of complex signal
    :param axis: axis of the signal, if multiple recevied signals specified
    """
    hb = _np.conj(_np.flipud(s))
    if axis < 0:
        axis += len(x.shape)
    padding = []
    x = _np.apply_along_axis(lambda a: _np.pad(a, (0, len(s)-1), 'constant'), axis, x)
    y = _sig.lfilter(hb, 1, x, axis=axis)
    y = _np.apply_along_axis(lambda a: a[len(s)-1:], axis, y)
    if not complex_output:
        y = _np.abs(y)
    return y

def lfilter0(b, a, x, axis=-1):
    """Filter data with an IIR or FIR filter with zero DC group delay.

    :func:`scipy.signal.lfilter` provides a way to filter a signal `x` using a FIR/IIR
    filter defined by `b` and `a`. The resulting output is delayed, as compared to the
    input by the group delay. This function corrects for the group delay, resulting in
    an output that is synchronized with the input signal. If the filter as an acausal
    impulse response, some precursor signal from the output will be lost. To avoid this,
    pad input signal `x` with sufficient zeros at the beginning to capture the precursor.
    Since both, :func:`scipy.signal.lfilter` and this function return a signal with the
    same length as the input, some signal tail is lost at the end. To avoid this, pad
    input signal `x` with sufficient zeros at the end.

    See documentation for :func:`scipy.signal.lfilter` for more details.

    >>> import arlpy
    >>> import numpy as np
    >>> fs = 250000
    >>> b = arlpy.uwa.absorption_filter(fs, distance=500)
    >>> x = np.pad(arlpy.signal.sweep(20000, 40000, 0.5, fs), (127, 127), 'constant')
    >>> y = arlpy.signal.lfilter0(b, 1, x)
    """
    w, g = _sig.group_delay((b, a))
    ndx = _np.argmin(_np.abs(w))
    d = int(round(g[ndx]))
    x = _np.apply_along_axis(lambda a: _np.pad(a, (0, d), 'constant'), axis, x)
    y = _sig.lfilter(b, a, x, axis)[d:]
    return y

def _lfilter_gen(b, a):
    x = _np.zeros(len(b))
    y = _np.zeros(len(a))
    while True:
        x = _np.roll(x, 1)
        x[0] = yield y[0]
        y = _np.roll(y, 1)
        y[0] = _np.sum(b*x) - _np.sum(a[1:]*y[1:])

def lfilter_gen(b, a):
    """Generator form of an FIR/IIR filter.

    The filter is a direct form I implementation of the standard difference
    equation. Data samples can be passed to the filter using the :func:`send`
    method, and the output can be read a sample at a time.

    >>> import arlpy
    >>> import numpy as np
    >>> import scipy.signal as sp
    >>> b, a = sp.iirfilter(2, 0.1, btype='lowpass')  # generate a biquad lowpass
    >>> f = arlpy.signal.filter_gen(b, a)             # create the filter
    >>> x = np.random.normal(0, 1, 1000)              # get some random data
    >>> y = [f.send(v) for v in x]                    # filter data by stepping through it
    """
    b = _np.asarray(b, dtype=_np.float)
    if not hasattr(a, "__len__") and a == 1:
        a = [1]
    a = _np.asarray(a, dtype=_np.float)
    if a[0] != 1.0:
        raise ValueError('a[0] must be 1')
    f = _lfilter_gen(b, a)
    f.__next__()
    return f

def nco_gen(fc, fs=2.0, phase0=0, wrap=2*_np.pi, func=lambda x: _np.exp(1j*x)):
    """Generator form of a numerically controlled oscillator (NCO).

    Samples at the output of the oscillator can be generated using the
    :func:`next` method. The oscillator frequency can be modified during
    operation using the :func:`send` method, with `fc` as the argument.

    If fs is specified, fc is given in Hz, otherwise it is specified as
    normalized frequency (Nyquist = 1).

    The default oscillator function is ``exp(i*phase)`` to generate a complex
    sinusoid. Alternate oscillator functions that take in the phase angle
    and generate other outputs can be specifed. For example, a real sinusoid
    can be generated by specifying ``sin`` as the function. The phase angle
    can be generated by specifying ``None`` as the function.

    :param fc: oscillation frequency
    :param fs: sampling frequency in Hz
    :param phase0: initial phase in radians (default: 0)
    :param wrap: phase angle to wrap phase around to 0 (default: :math:`2\\pi`)
    :param func: oscillator function of phase angle (default: complex sinusoid)

    >>> import arlpy
    >>> import math
    >>> nco = arlpy.signal.nco_gen(27000, 108000, func=math.sin)
    >>> x = [nco.next() for i in range(12)]
    >>> x = np.append(x, nco.send(54000))      # change oscillation frequency
    >>> x = np.append(x, [nco.next() for i in range(4)])
    >>> x
    [0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 1, -1, 1, -1, 1]
    """
    p = phase0
    while True:
        fc1 = yield p if func is None else func(p)
        if fc1 is not None:
            fc = fc1
        p = _np.mod(p + 2*_np.pi*fc/fs, wrap)

def nco(fc, fs=2.0, phase0=0, wrap=2*_np.pi, func=lambda x: _np.exp(1j*x)):
    """Numerically controlled oscillator (NCO).

    If fs is specified, fc is given in Hz, otherwise it is specified as
    normalized frequency (Nyquist = 1).

    The default oscillator function is ``exp(i*phase)`` to generate a complex
    sinusoid. Alternate oscillator functions that take in the phase angle
    and generate other outputs can be specifed. For example, a real sinusoid
    can be generated by specifying ``sin`` as the function. The phase angle
    can be generated by specifying ``None`` as the function.

    :param fc: array of instantaneous oscillation frequency
    :param fs: sampling frequency in Hz
    :param phase0: initial phase in radians (default: 0)
    :param wrap: phase angle to wrap phase around to 0 (default: :math:`2\\pi`)
    :param func: oscillator function of phase angle (default: complex sinusoid)

    >>> import arlpy
    >>> import numpy as np
    >>> fc = np.append([27000]*12, [54000]*5)
    >>> x = arlpy.signal.nco(fc, 108000, func=np.sin)
    >>> x
    [0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 1, -1, 1, -1, 1]
    """
    p = 2*_np.pi*fc/fs
    p[0] = phase0
    p = _np.mod(_np.cumsum(p), wrap)
    return p if func is None else func(p)

def correlate_periodic(a, v=None):
    """Cross-correlation of two 1-dimensional periodic sequences.

    a and v must be sequences with the same length. If v is not specified, it is
    assumed to be the same as a (i.e. the function computes auto-correlation).

    :param a: input sequence #1
    :param v: input sequence #2
    :returns: discrete periodic cross-correlation of a and v
    """
    a_fft = _np.fft.fft(_np.asarray(a))
    if v is None:
        v_cfft = a_fft.conj()
    else:
        v_cfft = _np.fft.fft(_np.asarray(v)).conj()
    x = _np.fft.ifft(a_fft * v_cfft)
    if _np.isrealobj(a) and (v is None or _np.isrealobj(v)):
        x = x.real
    return x

def goertzel(f, x, fs=2.0, filter=False):
    """Goertzel algorithm for single tone detection.

    The output of the Goertzel algorithm is the same as a single bin DFT if
    ``f/(fs/N)`` is an integer, where ``N`` is the number of points in signal ``x``.

    The detection metric returned by this function is the magnitude of the output
    of the Goertzel algorithm at the end of the input block. If ``filter`` is set
    to ``true``, the complex time series at the output of the IIR filter is returned,
    rather than just the detection metric.

    :param f: frequency of tone of interest in Hz
    :param x: real or complex input sequence
    :param fs: sampling frequency of x in Hz
    :param filter: output complex time series if true, detection metric otherwise (default: false)
    :returns: detection metric or complex time series

    >>> import arlpy
    >>> x1 = arlpy.signal.cw(64, 1, 512)
    >>> g1 = arlpy.signal.goertzel(64, x1, 512)
    >>> g1
    256.0
    >>> g2 = arlpy.signal.goertzel(32, x1, 512)
    >>> g2
    0.0
    """
    n = x.size
    m = f/(fs/n)
    if filter:
        y = _np.empty(n, dtype=_np.complex)
    w1 = 0
    w2 = 0
    for j in range(n):
        w0 = 2*_np.cos(2*_np.pi*m/n)*w1 - w2 + x[j]
        if filter:
            y[j] = w0 - _np.exp(-2j*_np.pi*m/n)*w1
        w2 = w1
        w1 = w0
    if filter:
        return y
    w0 = 2*_np.cos(2*_np.pi*m/n)*w1 - w2
    return _np.abs(w0 - _np.exp(-2j*_np.pi*m/n)*w1)
