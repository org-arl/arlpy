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
    """Compute short time Fourier transform of array data.

    :param x: time series data for multiple sensors (column per sensor)
    :param nfft: window length in samples
    :param overlap: number of samples of overlap between windows
    :param window: window function to use (None means rectangular window)
    :returns: 3d array of time x frequency x sensor

    For supported window functions, see documentation for :func:`scipy.signal.get_window`.
    """
    if overlap != 0:
        raise ValueError('Non-zero overlaps not yet implemented')  # TODO
    m, n = x.shape
    if m % nfft != 0:
        m = int(m/nfft)*nfft
    x = _np.reshape(x[:m, :], (-1, nfft, n))
    if window is not None:
        w = _sig.get_window(window, nfft)
        w = w[_np.newaxis, :, _np.newaxis]
        x *= w
    x = _np.fft.fft(x, axis=1)
    return x

def steering():
    # TODO
    pass

def _beamform(x, a, mode):
    # narrowband beamformer
    assert x.shape[1] == a.shape[1]
    # TODO

def beamform(x, a, mode=BARTLETT, out=POWER):
    """Beamform array data.

    For narrowband beamforming, the array data must be 2D with baseband time series for each
    sensor in individual columns. The steering vectors must also be 2D with a row per
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
    >>> y = bf.beamform(bf.stft(x, 256), bf.steering(TODO), mode=bf.CAPON)
    """
    if x.shape[1:] != a.shape[1:]:
        raise ValueError('Sensor count mismatch in data and steering vector')
    if mode != BARTLETT and mode != CAPON:
        raise ValueError('Invalid beamformer mode')
    if out != POWER and out != COMPLEX:
        raise ValueError('Invalid output type')
    # handle narrowband data
    if len(x.shape) == 2:
        bfo = _beamform(x, a, mode)
        return _np.abs(bfo)**2 if out == POWER else bfo
    # handle broadband data
    m, nfft, n = x.shape
    bfo = _np.empty((m, nfft, a.shape[0]))
    for f in range(nfft):
        bfo[:, f, :] = _beamform(x[:, f, :], a[:, f, :], mode)
    return _np.sum(_np.abs(bfo)**2, axis=1) if out == POWER else bfo
