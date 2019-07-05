##############################################################################
#
# Copyright (c) 2016-2019, Mandar Chitre
#
# This file is part of arlpy which is released under Simplified BSD License.
# See file LICENSE or go to http://www.opensource.org/licenses/BSD-3-Clause
# for full license details.
#
##############################################################################

"""Array signal processing / beamforming toolbox."""

import numpy as _np
import scipy.signal as _sig
import arlpy.plot as _plt
import arlpy.utils as _utils

def normalize(x, unit_variance=True):
    """Normalize array timeseries data to be zero-mean and equal variance.

    The average signal power across the array is retained if `unit_variance`
    is set to True so that the beamformed data can be compared with other datsets.

    :param x: passband real timeseries data for multiple sensors (row per sensor)
    :param unit_variance: True to make timeseries unit variance,
                          False to retain average signal power across the array
    :returns: normalized passband real timeseries data
    """
    m = _np.mean(x, axis=-1, keepdims=True)
    v = _np.var(x, axis=-1, keepdims=True)
    s = 1.0 if unit_variance else _np.sqrt(v.mean())
    v[v == 0] = 1
    return s*(x-m)/_np.sqrt(v)

def stft(x, nfft, overlap=0, window=None):
    """Compute short time Fourier transform (STFT) of array data.

    :param x: passband real timeseries data for multiple sensors (row per sensor)
    :param nfft: window length in samples
    :param overlap: number of samples of overlap between windows
    :param window: window function to use (None means rectangular window)
    :returns: 3d array of time x frequency x sensor

    For supported window functions, see documentation for :func:`scipy.signal.get_window`.
    """
    n, m = x.shape
    if overlap == 0:
        if m % nfft != 0:
            m = (m//nfft)*nfft
        x = _np.reshape(x[:,:m], (n, -1, nfft))
    elif overlap > 0 and overlap < nfft:
        p = (m-overlap)//(nfft-overlap)
        y = _np.empty((n, p, nfft))
        for j in range(p):
            y[:,j,:] = x[:,(nfft-overlap)*j:(nfft-overlap)*j+nfft]
        x = y
    else:
        raise ValueError('overlap must be in the range [0,nfft)')
    if window is not None:
        x *= _sig.get_window(window, nfft)
    x = _np.fft.fft(x, axis=-1)
    return x

def steering_plane_wave(pos, c, theta):
    """Compute steering delays assuming incoming signal has a plane wavefront.

    For linear arrays, pos is 1D array. For planar and 3D arrays, pos is a 2D array with a
    sensor position vector in each row.

    For linear arrays, theta is a 1D array of angles (in radians) with 0 being broadside. For
    planar and 3D arrays, theta is a 2D array with an (azimuth, elevation) pair in each row.
    Such arrays can be easily generated using the :func:`arlpy.utils.linspace2d` function.

    The broadside direction is along the x-axis of a right-handed coordinate system with z-axis pointing
    upwards, and has azimuth and elevation as 0. In case of linear arrays, the y-coordinate is the
    sensor position. In case of planar arrays, if only 2 coordinates are provided, these coordinates
    are assumed to be y and z.

    :param pos: sensor positions (m)
    :param c: signal propagation speed (m/s)
    :param theta: steering directions (radians)
    :returns: steering delays with a row for each direction (s)

    >>> import numpy as np
    >>> from arlpy import bf, utils
    >>> pos1 = [0.0, 0.5, 1.0, 1.5, 2.0]
    >>> a1 = bf.steering_plane_wave(pos1, 1500, np.deg2rad(np.linspace(-90, 90, 181)))
    >>> pos2 = [[0.0, 0.0],
                [0.0, 0.5],
                [0.5, 0.0],
                [0.5, 0.5]]
    >>> a2 = bf.steering_plane_wave(pos2, 1500, np.deg2rad(utils.linspace2d(-20, 20, 41, -10, 10, 21)))
    """
    pos = _np.array(pos, dtype=_np.float)
    theta = _np.asarray(theta, dtype=_np.float)
    if pos.ndim == 1:
        pos -= _np.mean(pos)
        dist = pos[:,_np.newaxis] * _np.sin(theta)
    else:
        if pos.shape[1] != 2 and pos.shape[1] != 3:
            raise ValueError('Sensor positions must be either 2d or 3d vectors')
        pos -= _np.mean(pos, axis=0)
        if pos.shape[1] == 2:
            pos = _np.c_[_np.zeros(pos.shape[0]), pos]
        azim = theta[:,0]
        elev = theta[:,1]
        dvec = _np.array([_np.cos(elev)*_np.cos(azim), _np.cos(elev)*_np.sin(azim), _np.sin(elev)])
        dist = _np.dot(pos, dvec)
    return -dist.T/c

def delay_and_sum(x, fs, sd, shading=None):
    """Time-domain delay-and-sum beamformer.

    The array data must be 2D with timeseries for each sensor in
    individual rows. The steering delays must also be 2D with a row per
    steering direction.

    :param x: passband real timeseries data for multiple sensors (row per sensor)
    :param fs: sampling rate for the array data (Hz)
    :param sd: steering delays (s)
    :param shading: window function to use for array shading (None means no shading)
    :returns: beamformer timeseries output with time as the last axis, and
              steering directions as the first

    >>> from arlpy import bf
    >>> import numpy as np
    >>> # timeseries array data assumed to be loaded in x
    >>> # sensor positions assumed to be in pos
    >>> y = bf.delay_and_sum(x, 1000, bf.steering(pos, np.linspace(-np.pi/2, np.pi/2, 181)))
    """
    if x.shape[0] != sd.shape[1]:
        raise ValueError('Sensor count mismatch in data and steering vector')
    if shading is None:
        s = _np.ones(sd.shape[1])
    else:
        s = _sig.get_window(shading, sd.shape[1])
        s /= _np.sqrt(_np.mean(s**2))
    bfo = _np.zeros((sd.shape[0], x.shape[1]))
    left = right = 0
    for j in range(sd.shape[0]):
        for k in range(sd.shape[1]):
            d = -int(_np.rint(sd[j,k]*fs))
            left = max(left, d)
            right = max(right, -d)
            bfo[j] += s[k]*_np.roll(x[k], d)
    return bfo[:,left:-right] if right > 0 else bfo[:,left:]

def covariance(x):
    """Compute array covariance matrix.

    :param x: narrowband complex timeseries data for multiple sensors (row per sensor)
    """
    cov_mtx = _np.zeros((x.shape[0], x.shape[0]), dtype=_np.complex)
    for j in range(x.shape[1]):
        cov_mtx += _np.outer(x[:,j], x[:,j].conj())
    cov_mtx /= x.shape[1]
    return cov_mtx

def bartlett(x, fc, sd, shading=None, complex_output=False):
    """Frequency-domain Bartlett beamformer.

    The timeseries data must be 2D with narrowband complex timeseries for each sensor in
    individual rows. The steering delays must also be 2D with a row per steering direction.

    If the timeseries data is specified as 1D array, it is assumed to represent multiple sensors
    at a single time.

    :param x: narrowband complex timeseries data for multiple sensors (row per sensor)
    :param fc: carrier frequency for the array data (Hz)
    :param sd: steering delays (s)
    :param shading: window function to use for array shading (None means no shading)
    :param complex_output: True for complex signal, False for beamformed power
    :returns: beamformer output averaged across time

    >>> from arlpy import bf
    >>> import numpy as np
    >>> # narrowband (1 kHz) timeseries array data assumed to be loaded in x
    >>> # sensor positions assumed to be in pos
    >>> y = bf.bartlett(x, 1000, bf.steering(pos, 1500, np.linspace(-np.pi/2, np.pi/2, 181)))
    """
    if x.ndim == 1:
        x = x[:,_np.newaxis]
    if x.shape[0] != sd.shape[1]:
        raise ValueError('Sensor count mismatch in data and steering vector')
    if fc == 0:
        a = _np.ones_like(sd)
    else:
        a = _np.exp(-2j*_np.pi*fc*sd)/_np.sqrt(sd.shape[1])
    if shading is not None:
        s = _sig.get_window(shading, a.shape[1])
        a *= s/_np.sqrt(_np.mean(s**2))
    if complex_output:
        return a.conj().dot(x)
    else:
        R = covariance(x)
        return _np.array([a[j].conj().dot(R).dot(a[j]).real for j in range(a.shape[0])])
        #return _np.array([1.0/a[j].conj().dot(_np.linalg.inv(R)).dot(a[j]).real for j in range(a.shape[0])])

def bartlett_beampattern(i, fc, sd, shading=None, theta=None, show=False):
    """Computes the beampattern for a Bartlett or delay-and-sum beamformer.

    :param i: row index of target steering distances
    :param fc: carrier frequency for the array data (Hz)
    :param sd: steering delays (s)
    :param shading: window function to use for array shading (None means no shading)
    :param theta: angles (in radians) for display if beampattern is plotted
    :param show: True to plot the beampattern, False to return it
    :returns: beampattern power response at all directions corresponding to rows in sd

    >>> from arlpy import bf
    >>> import numpy as np
    >>> sd = bf.steering(np.linspace(0, 5, 11), 1500, np.linspace(-np.pi/2, np.pi/2, 181))
    >>> bp = bf.bartlett_beampattern(90, 1500, sd, show=True)
    """
    a = _np.exp(-2j*_np.pi*fc*sd)/_np.sqrt(sd.shape[1])
    if shading is not None:
        s = _sig.get_window(shading, a.shape[1])
        a *= s/_np.sqrt(_np.mean(s**2))
    bp = _np.abs(_np.dot(a.conj(), a[i]))**2
    if show:
        if theta is None:
            _plt.plot(_utils.pow2db(bp), ylabel='Array response (dB)', title='Beam #'+str(i))
        else:
            a = theta * 180/_np.pi
            _plt.plot(a, _utils.pow2db(bp), xlabel='Angle (deg)', ylabel='Array response (dB)', title='Beam #%d @ %0.1f deg'%(i, a[i]))
    else:
        return bp

def capon(x, fc, sd, complex_output=False):
    """Frequency-domain Capon beamformer.

    The timeseries data must be 2D with narrowband complex timeseries for each sensor in
    individual rows. The steering delays must also be 2D with a row per steering direction.

    If the timeseries data is specified as 1D array, it is assumed to represent multiple sensors
    at a single time.

    The covariance matrix of x is estimated over the entire timeseries, and used to compute
    the optimal weights for the Capon beamformer.

    :param x: narrowband complex timeseries data for multiple sensors (row per sensor)
    :param fc: carrier frequency for the array data (Hz)
    :param sd: steering delays (s)
    :param complex_output: True for complex signal, False for beamformed power
    :returns: beamformer output averaged across time

    >>> from arlpy import bf
    >>> import numpy as np
    >>> # narrowband (1 kHz) timeseries array data assumed to be loaded in x
    >>> # sensor positions assumed to be in pos
    >>> y = bf.capon(x, 1000, bf.steering(pos, 1500, np.linspace(-np.pi/2, np.pi/2, 181)))
    """
    if x.ndim == 1:
        x = x[:,_np.newaxis]
    if x.shape[0] != sd.shape[1]:
        raise ValueError('Sensor count mismatch in data and steering vector')
    if fc == 0:
        a = _np.ones_like(sd)
    else:
        a = _np.exp(-2j*_np.pi*fc*sd)/_np.sqrt(sd.shape[1])
    if complex_output:
        R = covariance(x)
        if _np.linalg.cond(R) > 10000:
            R += _np.random.normal(0, _np.max(_np.abs(R))/1000000, R.shape)
        Rinv = _np.linalg.inv(R)
        w = _np.array([Rinv.dot(a[j])/(a[j].conj().dot(Rinv).dot(a[j])) for j in range(a.shape[0])])
        return w.conj().dot(x)
    else:
        R = covariance(x)
        if _np.linalg.cond(R) > 10000:
            R += _np.random.normal(0, _np.max(_np.abs(R))/1000000, R.shape)
        return _np.array([1.0/a[j].conj().dot(_np.linalg.inv(R)).dot(a[j]).real for j in range(a.shape[0])])

def broadband(x, fs, nfft, sd, f0=0, fmin=None, fmax=None, overlap=0, beamformer=bartlett):
    """Frequency-domain broadband beamformer operating on time-domain input data.

    The broadband beamformer is implementing by taking STFT of the data, applying narrowband
    beamforming to each frequency bin, and integrating the beamformer output power across
    the entire bandwidth.

    The array data must be 2D with timeseries for each sensor in individual rows. The steering
    delays must also be 2D with a row per steering direction.

    The STFT window size should be chosen such that the corresponding distance (based on wave
    propagation speed) is much larger than the aperture size of the array.

    If the array data is real and f0 is zero, the data is assumed to be passband and so only
    half the frequency components are computed.

    :param x: timeseries data for multiple sensors (row per sensor)
    :param fs: sampling rate for array data (Hz)
    :param c: wave propagation speed (m/s)
    :param nfft: STFT window size
    :param sd: steering distances (m)
    :param f0: carrier frequency (for baseband data) (Hz)
    :param fmin: minimum frequency to integrate (Hz)
    :param fmax: maximum frequency to integrate (Hz)
    :param overlap: window overlap for STFT
    :param beamformer: narrowband beamformer to use
    :returns: beamformer output with steering directions as the first axis, time as the second,
              and if complex output, fft bins as the third

    >>> from arlpy import bf
    >>> # passband timeseries array data assumed to be loaded in x, sampled at fs
    >>> # sensor positions assumed to be in pos
    >>> sd = bf.steering(pos, 1500, np.linspace(-np.pi/2, np.pi/2, 181))
    >>> y = bf.broadband(x, fs, 256, sd, beamformer=capon)
    """
    if nfft/fs < (_np.max(sd)-_np.min(sd)):
        raise ValueError('nfft too small for this array')
    nyq = 2 if f0 == 0 and _np.sum(_np.abs(x.imag)) == 0 else 1
    x = stft(x, nfft, overlap)
    bfo = _np.zeros((sd.shape[0], x.shape[1], nfft//nyq), dtype=_np.complex)
    for i in range(nfft//nyq):
        f = i if i < nfft/2 else i-nfft
        f = f0 + f*float(fs)/nfft
        if (fmin is None or f >= fmin) and (fmax is None or f <= fmax):
            bfo[:,:,i] = nyq*beamformer(x[:,:,i], f, sd, complex_output=True)
    return (_np.abs(bfo)**2).sum(axis=-1)
