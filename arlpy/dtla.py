"""DTLA support toolbox."""

import os as _os
import numpy as _np
from scipy import signal as _sig

_fs = 1/(1.6e-6*26)
_framelen = 2*26
_channels = 24
_magic = 0xc0de

def check(filename):
    """Check if a file is likely to be a valid DTLA datafile."""
    statinfo = _os.stat(filename)
    if statinfo.st_size >= 2*2*_channels:
        with open(filename, 'rb') as f:
            data = _np.fromfile(f, dtype=_np.uint16, count=_framelen/2)
        if data[0] == _magic & data[1] == _magic:
            return True
    return False

def get_sampling_rate(filename=None):
    """Get the sampling rate."""
    return _fs

def get_channels(filename=None):
    """Get the number of available data channels."""
    return _channels

def get_data_length(filename):
    """Get the length of the datafile in samples."""
    statinfo = _os.stat(filename)
    return statinfo.st_size//_framelen

def get_data(filename, channel=None, start=0, length=None, detrend='linear'):
    """Load selected data from DTLA recording."""
    if channel is None:
        channel = range(_channels)
    elif isinstance(channel, int):
        channel = [channel]
    if length is None:
        length = get_data_length(filename)-start
    with open(filename, 'rb') as f:
        f.seek(start*_framelen, _os.SEEK_SET)
        data = _np.fromfile(f, dtype=_np.uint16, count=_framelen/2*length)
    data = _np.reshape(data, [length,_framelen/2])
    data = data[:,2:]
    data = _np.take(data, channel, axis=1).astype(_np.float)
    if len(channel) == 1:
        data = data.ravel()
    data = 5*data/65536-2.5
    if detrend is not None:
        data = _sig.detrend(data, axis=0, type=detrend)
    return data
