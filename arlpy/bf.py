##############################################################################
#
# Copyright (c) 2016, Mandar Chitre
#
# This file is part of arlpy which is released under Simplified BSD License.
# See file LICENSE or go to http://www.opensource.org/licenses/BSD-3-Clause
# for full license details.
#
##############################################################################

"""Array signal processing / beamforming toolbox."""

import numpy as _np
import scipy.signal as _sig

def normalize(x):
    """Normalize array time series data to be zero-mean and unit variance.

    :param x: time series data for multiple sensors (column per sensor)
    :returns: normalized time series data
    """
    m = _np.mean(x, axis=0, keepdims=True)
    v = _np.var(x, axis=0, keepdims=True)
    v[v == 0] = 1
    return (x-m)/_np.sqrt(v)

def stft(x, nfft, overlap=0, window=None):
    """Compute short time Fourier transform (STFT) of array data.

    :param x: time series data for multiple sensors (column per sensor)
    :param nfft: window length in samples
    :param overlap: number of samples of overlap between windows
    :param window: window function to use (None means rectangular window)
    :returns: 3d array of time x frequency x sensor

    For supported window functions, see documentation for :func:`scipy.signal.get_window`.
    """
    if overlap != 0:
        raise ValueError('Non-zero overlaps not implemented')  # TODO
    m, n = x.shape
    if m % nfft != 0:
        m = int(m/nfft)*nfft
    x = _np.reshape(x[:m,:], (-1, nfft, n))
    if window is not None:
        w = _sig.get_window(window, nfft)
        w = w[_np.newaxis,:,_np.newaxis]
        x *= w
    x = _np.fft.fft(x, axis=1)
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
    :returns: steering distances (m) with a column for each direction

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
    pos = _np.asarray(pos, dtype=_np.float)
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
    return -dist

def bartlett(x, fc, c, sd, shading=None, complex_output=False):
    """Bartlett beamformer.

    The array data must be 2D with baseband time series for each sensor in
    individual columns. The steering vectors must also be 2D with a column per
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
    :returns: beamformer output with time as the first axis, and steering directions as the other

    >>> from arlpy import bf
    >>> import numpy as np
    >>> # narrowband (1 kHz) time series array data assumed to be loaded in x
    >>> # sensor positions assumed to be in pos
    >>> y = bf.bartlett(x, 1000, 1500, bf.steering(pos, np.linspace(-np.pi/2, np.pi/2, 181)))
    """
    if x.ndim == 1:
        x = x[_np.newaxis,:]
    if x.shape[1] != sd.shape[0]:
        raise ValueError('Sensor count mismatch in data and steering vector')
    if fc == 0:
        a = _np.ones_like(sd)
    else:
        wavelength = float(c)/fc
        a = _np.exp(-2j*_np.pi*sd/wavelength)/_np.sqrt(sd.shape[0])
    if shading is not None:
        s = _sig.get_window(shading, a.shape[0])
        a *= s[:,_np.newaxis]/_np.sqrt(_np.mean(s**2))
    bfo = _np.dot(x, a.conj())
    return bfo if complex_output else _np.abs(bfo)**2

def capon(x, fc, c, sd, complex_output=False):
    """Capon beamformer.

    The array data must be 2D with baseband time series for each sensor in
    individual columns. The steering vectors must also be 2D with a column per
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
        x = x[_np.newaxis,:]
    if x.shape[1] != sd.shape[0]:
        raise ValueError('Sensor count mismatch in data and steering vector')
    if fc == 0:
        w = _np.ones_like(sd)
    else:
        wavelength = float(c)/fc
        a = _np.exp(-2j*_np.pi*sd/wavelength)/_np.sqrt(sd.shape[0])
        # TODO compute w
    bfo = _np.dot(x, w.conj())
    return bfo if complex_output else _np.abs(bfo)**2

def broadband(x, fs, c, nfft, sd, f0=0, beamformer=bartlett, complex_output=False):
    """Broadband beamformer.

    The broadband beamformer is implementing by taking STFT of the data, applying narrowband
    beamforming to each frequency bin, and integrating the beamformer output power across
    the entire bandwidth.

    The array data must be 2D with baseband time series for each sensor in
    individual columns. The steering vectors must also be 2D with a column per
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
    :returns: beamformer output with time as the first axis, and steering directions as the other

    >>> from arlpy import bf
    >>> # passband time series array data assumed to be loaded in x, sampled at fs
    >>> # sensor positions assumed to be in pos
    >>> sd = bf.steering(pos, np.linspace(-np.pi/2, np.pi/2, 181))
    >>> y = bf.broadband(x, fs, 256, sd, beamformer=capon)
    """
    nyq = 2 if f0 == 0 and _np.sum(_np.abs(x.imag)) == 0 else 1
    x = stft(x, nfft)
    bfo = _np.zeros((x.shape[0], sd.shape[1], nfft/nyq), dtype=_np.complex if complex_output else _np.float)
    for i in range(nfft/nyq):
        f = i if i < nfft/2 else i-nfft
        f = f0 + f*float(fs)/nfft
        bfo[:,:,i] = beamformer(x[:,i,:], f, c, sd, complex_output=complex_output)
    return bfo if complex_output else _np.sum(bfo, axis=2)
