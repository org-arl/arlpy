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

def normalize(x):
    """Normalize array time series data to be zero-mean and unit variance.

    :param x: time series data for multiple sensors (row per sensor)
    :returns: normalized time series data
    """
    m = _np.mean(x, axis=-1, keepdims=True)
    v = _np.var(x, axis=-1, keepdims=True)
    v[v == 0] = 1
    return (x-m)/_np.sqrt(v)

def stft(x, nfft, overlap=0, window=None):
    """Compute short time Fourier transform (STFT) of array data.

    :param x: time series data for multiple sensors (row per sensor)
    :param nfft: window length in samples
    :param overlap: number of samples of overlap between windows
    :param window: window function to use (None means rectangular window)
    :returns: 3d array of time x frequency x sensor

    For supported window functions, see documentation for :func:`scipy.signal.get_window`.
    """
    if overlap != 0:
        raise ValueError('Non-zero overlaps not implemented')  # TODO
    n, m = x.shape
    if m % nfft != 0:
        m = (m//nfft)*nfft
    x = _np.reshape(x[:,:m], (n, -1, nfft))
    if window is not None:
        x *= _sig.get_window(window, nfft)
    x = _np.fft.fft(x, axis=-1)
    return x

def steering(pos, theta):
    """Compute steering distances.

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
    :param theta: steering directions (radians)
    :returns: steering distances (m) with a row for each direction

    >>> import numpy as np
    >>> from arlpy import bf, utils
    >>> pos1 = [0.0, 0.5, 1.0, 1.5, 2.0]
    >>> a1 = bf.steering(pos1, np.deg2rad(np.linspace(-90, 90, 181)))
    >>> pos2 = [[0.0, 0.0],
                [0.0, 0.5],
                [0.5, 0.0],
                [0.5, 0.5]]
    >>> a2 = bf.steering(pos2, np.deg2rad(utils.linspace2d(-20, 20, 41, -10, 10, 21)))
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
    return -dist.T

def bartlett(x, fc, c, sd, shading=None, complex_output=False):
    """Bartlett beamformer.

    The array data must be 2D with baseband time series for each sensor in
    individual rows. The steering vectors must also be 2D with a row per
    steering direction, as produced by the :func:`steering` function.

    If the array data is specified as 1D array, it is assumed to represent multiple sensors
    at a single time.

    The array data is assumed to be narrowband. If broadband beamforming is desired, use the
    :func:`broadband` function instead.

    :param x: array data
    :param fc: carrier frequency for the array data (Hz)
    :param c: wave propagation speed (m/s)
    :param sd: steering distances (m)
    :param shading: window function to use for array shading (None means no shading)
    :param complex_output: True for complex signal, False for beamformed power
    :returns: beamformer output with time as the last axis, and steering directions as the first

    >>> from arlpy import bf
    >>> import numpy as np
    >>> # narrowband (1 kHz) time series array data assumed to be loaded in x
    >>> # sensor positions assumed to be in pos
    >>> y = bf.bartlett(x, 1000, 1500, bf.steering(pos, np.linspace(-np.pi/2, np.pi/2, 181)))
    """
    if x.ndim == 1:
        x = x[:,_np.newaxis]
    if x.shape[0] != sd.shape[1]:
        raise ValueError('Sensor count mismatch in data and steering vector')
    if fc == 0:
        a = _np.ones_like(sd)
    else:
        wavelength = float(c)/fc
        a = _np.exp(-2j*_np.pi*sd/wavelength)/_np.sqrt(sd.shape[1])
    if shading is not None:
        s = _sig.get_window(shading, a.shape[1])
        a *= s/_np.sqrt(_np.mean(s**2))
    bfo = _np.dot(a.conj(), x)
    return bfo if complex_output else _np.abs(bfo)**2

def bartlett_beampattern(i, fc, c, sd, shading=None, theta=None, show=False):
    """Computes the beampattern for a Bartlett beamformer.

    :param i: row index of target steering distances
    :param fc: carrier frequency for the array data (Hz)
    :param c: wave propagation speed (m/s)
    :param sd: steering distances (m)
    :param shading: window function to use for array shading (None means no shading)
    :param theta: angles (in radians) for display if beampattern is plotted
    :param show: True to plot the beampattern, False to return it
    :returns: beampattern power response at all directions corresponding to rows in sd

    >>> from arlpy import bf
    >>> import numpy as np
    >>> sd = bf.steering(np.linspace(0, 5, 11), np.linspace(-np.pi/2, np.pi/2, 181))
    >>> bp = bf.bartlett_beampattern(90, 1500, 1500, sd, show=True)
    """
    wavelength = float(c)/fc
    a = _np.exp(-2j*_np.pi*sd/wavelength)/_np.sqrt(sd.shape[1])
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

def capon(x, fc, c, sd, complex_output=False):
    """Capon beamformer.

    The array data must be 2D with baseband time series for each sensor in
    individual rows. The steering vectors must also be 2D with a row per
    steering direction, as produced by the :func:`steering` function.

    If the array data is specified as 1D array, it is assumed to represent multiple sensors
    at a single time.

    The array data is assumed to be narrowband. If broadband beamforming is desired, use the
    :func:`broadband` function instead.

    :param x: array data
    :param fc: carrier frequency for the array data (Hz)
    :param c: wave propagation speed (m/s)
    :param sd: steering distances (m)
    :param complex_output: True for complex signal, False for beamformed power
    :returns: beamformer output with time as the first axis, and steering directions as the other

    >>> from arlpy import bf
    >>> import numpy as np
    >>> # narrowband (1 kHz) time series array data assumed to be loaded in x
    >>> # sensor positions assumed to be in pos
    >>> y = bf.capon(x, 1000, 1500, bf.steering(pos, np.linspace(-np.pi/2, np.pi/2, 181)))
    """
    if x.ndim == 1:
        x = x[:,_np.newaxis]
    if x.shape[0] != sd.shape[1]:
        raise ValueError('Sensor count mismatch in data and steering vector')
    if fc == 0:
        a = _np.ones_like(sd)
    else:
        wavelength = float(c)/fc
        a = _np.exp(-2j*_np.pi*sd/wavelength)/_np.sqrt(sd.shape[1])
    # TODO compute w and multiply with a
    raise ValueError('Capon not implemented yet')
    bfo = _np.dot(a.conj(), x)
    return bfo if complex_output else _np.abs(bfo)**2

def broadband(x, fs, c, nfft, sd, f0=0, beamformer=bartlett, complex_output=False):
    """Broadband beamformer.

    The broadband beamformer is implementing by taking STFT of the data, applying narrowband
    beamforming to each frequency bin, and integrating the beamformer output power across
    the entire bandwidth.

    The array data must be 2D with baseband time series for each sensor in
    individual rows. The steering vectors must also be 2D with a row per
    steering direction, as produced by the :func:`steering` function.

    The STFT window size should be chosen such that the corresponding distance (based on wave
    propagation speed) is much larger than the aperture size of the array.

    If the array data is real and f0 is zero, the data is assumed to be passband and so only
    half the frequency components are computed.

    :param x: array data
    :param fs: sampling rate for array data (Hz)
    :param c: wave propagation speed (m/s)
    :param nfft: STFT window size
    :param sd: steering distances (m)
    :param f0: carrier frequency (for baseband data) (Hz)
    :param beamformer: narrowband beamformer to use
    :param complex_output: True for complex signal, False for beamformed power
    :returns: beamformer output with steering directions as the first axis, time as the second,
              and if complex output, fft bins as the third

    >>> from arlpy import bf
    >>> # passband time series array data assumed to be loaded in x, sampled at fs
    >>> # sensor positions assumed to be in pos
    >>> sd = bf.steering(pos, np.linspace(-np.pi/2, np.pi/2, 181))
    >>> y = bf.broadband(x, fs, 256, sd, beamformer=capon)
    """
    if nfft/fs < (_np.max(sd)-_np.min(sd))/c:
        raise ValueError('nfft too small for this array')
    nyq = 2 if f0 == 0 and _np.sum(_np.abs(x.imag)) == 0 else 1
    x = stft(x, nfft)
    bfo = _np.zeros((sd.shape[0], x.shape[1], nfft//nyq), dtype=_np.complex if complex_output else _np.float)
    for i in range(nfft//nyq):
        f = i if i < nfft/2 else i-nfft
        f = f0 + f*float(fs)/nfft
        bfo[:,:,i] = nyq*beamformer(x[:,:,i], f, c, sd, complex_output=complex_output)
    return bfo if complex_output else _np.sum(bfo, axis=-1)
