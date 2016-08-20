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

# beamformer mode constants
BARTLETT = 'bartlett'
CAPON = 'capon'

# beamformer output constants
POWER = 'power'

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

def steer(pos, theta, wavelength=2, nfft=1, fs=None, fc=None):
    """Generate steering vectors.

    For linear arrays, pos is 1D array. For planar and 3D arrays, pos is a 2D array with a
    sensor position vector in each row.

    For linear arrays, theta is a 1D array of angles (in radians) with 0 being broadside. For
    planar and 3D arrays, theta is a 2D array with an (azimuth, elevation) pair in each row.
    Such arrays can be easily generated using the :func:`arlpy.utils.linspace2d` function.
    The broadside direction is along the z-axis, and has azimuth and elevation as 0.

    If the wavelength is not specified, it is assumed that the position vectors are normalized
    to units of half-wavelength.

    If nfft > 1 then fs and fc must be specified for broadband beamforming.

    :param pos: sensor positions
    :param theta: steering directions
    :param wavelength: wavelength at carrier frequency
    :param nfft: STFT window length in samples
    :param fs: sampling rate
    :param fc: carrier frequency
    :returns: steering column vectors for each direction

    >>> import numpy as np
    >>> from arlpy import bf, utils
    >>> pos1 = [0.0, 0.5, 1.0, 1.5, 2.0]
    >>> a1 = bf.steer(pos1, np.deg2rad(np.linspace(-90, 90, 181)))
    >>> pos2 = [[0.0, 0.0],
                [0.0, 0.5],
                [0.5, 0.0],
                [0.5, 0.5]]
    >>> a2 = bf.steer(pos2, np.deg2rad(utils.linspace2d(-20, 20, 41, -10, 10, 21)))
    """
    pos = _np.asarray(pos, dtype=_np.float)
    pos /= wavelength/2.0
    theta = _np.asarray(theta, dtype=_np.float)
    if pos.ndim == 1:
        pos -= _np.mean(pos)
        phase = _np.pi * pos[:,_np.newaxis] * _np.sin(theta)
    else:
        if pos.shape[1] != 2 and pos.shape[1] != 3:
            raise ValueError('Sensor positions must be either 2d or 3d vectors')
        pos -= _np.mean(pos, axis=0)
        if pos.shape[1] == 2:
            pos = _np.c_[pos, _np.zeros(pos.shape[0])]
        azim = theta[:,0]
        elev = theta[:,1]
        dvec = _np.array([-_np.sin(azim)*_np.cos(elev), _np.sin(elev), _np.cos(azim)*_np.cos(elev)])
        phase = _np.pi * _np.dot(pos, dvec)
    if nfft > 1:
        phase2 = _np.empty((phase.shape[0], nfft, phase.shape[1]), dtype=_np.complex)
        for i in range(nfft):
            phase2[:,i,:] = phase * (1+float(i*fs)/(nfft*fc))
        phase = phase2
    return _np.exp(-1j * phase)/_np.sqrt(pos.shape[0])

def _beamform(x, a, mode):
    # narrowband beamformer
    assert x.shape[1] == a.shape[1]
    # TODO

def beamform(x, a, mode=BARTLETT, out=POWER):
    """Beamform array data.

    For narrowband beamforming, the array data must be 2D with baseband time series for each
    sensor in individual columns. The steering vectors must also be 2D with a column per
    steering direction, as produced by the :func:`steering` function.

    For broadband beamforming, the array data must be 3D with time, frequency and sensors as
    the dimensions, as produced by the :func:`stft` function. The steering vectors must also be
    3D, as produced by the :func:`steering` function.

    :param x: array data
    :param a: steering vectors
    :param mode: beamformer mode (BARTLETT or CAPON)
    :param out: output type (POWER or COMPLEX)
    :returns: beamformer output with time as the first axis, and steering directions as the last axis

    In case of broadband beamforming with COMPLEX output, the beamformer output has frequency as
    the middle axis. In case of broadband beamforming with POWER output, the power is integrated
    across the entire bandwidth.

    >>> from arlpy import bf
    >>> # broadband time series array data assumed to be loaded in x
    >>> # with sampling rate fs and carrier frequency fc
    >>> # sensor positions assumed to be in pos
    >>> a = bf.steering(pos, np.linspace(-np.pi/2, np.pi/2, 181), nfft=256, fs=fs, fc=fc)
    >>> y = bf.beamform(bf.stft(x, 256), a, mode=bf.CAPON)
    """
    if mode != BARTLETT and mode != CAPON:
        raise ValueError('Invalid beamformer mode')
    if out != POWER and out != COMPLEX:
        raise ValueError('Invalid output type')
    if x.shape[-1] != a.shape[0]:
        raise ValueError('Sensor count mismatch in data and steering vector')
    # handle narrowband data
    if len(x.shape) == 2:
        bfo = _beamform(x, a, mode)
        return _np.abs(bfo)**2 if out == POWER else bfo
    # handle broadband data
    m, nfft, n = x.shape
    if nfft != a.shape[1]:
        raise ValueError('Frequency bin mismatch in data and steering vector')
    bfo = _np.empty((m, nfft, a.shape[0]))
    for f in range(nfft):
        bfo[:,f,:] = _beamform(x[:,f,:], a[:,f,:], mode)
    return _np.sum(_np.abs(bfo)**2, axis=1) if out == POWER else bfo
