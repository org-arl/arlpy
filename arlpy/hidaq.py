##############################################################################
#
# Copyright (c) 2016, Mandar Chitre
#
# This file is part of arlpy which is released under Simplified BSD License.
# See file LICENSE or go to http://www.opensource.org/licenses/BSD-3-Clause
# for full license details.
#
##############################################################################

"""HiDAQ support toolbox."""

import os as _os
import numpy as _np
from scipy import signal as _sig

_fs = 500000
_channels = 4
_framelen = 8

def check(filename):
    """Check if a file is likely to be a valid HiDAQ datafile."""
    statinfo = _os.stat(filename)
    if statinfo.st_size >= 4:
        with open(filename, 'rb') as f:
            hdr = _np.fromfile(f, dtype=_np.dtype('>i4'), count=1)
        if _np.round(hdr[0]/66.0) == _channels:
            return True
    return False

def get_sampling_rate(filename=None):
    """Get the sampling rate in Hz."""
    return _fs

def get_channels(filename=None):
    """Get the number of available data channels."""
    return _channels

def get_data_length(filename):
    """Get the length of the datafile in samples."""
    statinfo = _os.stat(filename)
    if statinfo.st_size < 4:
        return 0
    with open(filename, 'rb') as f:
        hdr = _np.fromfile(f, dtype=_np.dtype('>i4'), count=1)
    return (statinfo.st_size-hdr[0])//_framelen

def get_data(filename, channel=None, start=0, length=None, detrend=None):
    """Load selected data from HiDAQ recording.

    :param filename: name of the datafile
    :param channel: list of channels to read (base 0, None to read all channels)
    :param start: sample index to start from
    :param length: number of samples to read (None means read all available samples)
    :param detrend: processing to be applied to each channel to remove offset/bias
                    (supported values: ``'linear'``, ``'constant'``, ``None``)
    """
    if channel is None:
        channel = range(_channels)
    elif isinstance(channel, int):
        channel = [channel]
    if length is None:
        length = get_data_length(filename)-start
    with open(filename, 'rb') as f:
        hdr = _np.fromfile(f, dtype=_np.dtype('>i4'), count=1)
        f.seek(hdr[0]+start*_framelen, _os.SEEK_SET)
        data = _np.fromfile(f, dtype=_np.dtype('>i2'), count=_channels*length)
    data = _np.reshape(data, [length,_channels])
    data = _np.take(data, channel, axis=1).astype(_np.float)
    if len(channel) == 1:
        data = data.ravel()
    data = data/2048
    if detrend is not None:
        data = _sig.detrend(data, axis=0, type=detrend)
    return data
