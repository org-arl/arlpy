##############################################################################
#
# Copyright (c) 2018, Mandar Chitre
#
# This file is part of arlpy which is released under Simplified BSD License.
# See file LICENSE or go to http://www.opensource.org/licenses/BSD-3-Clause
# for full license details.
#
##############################################################################

"""UNET support toolbox."""

import re as _re
import base64 as _b64
import struct as _struct
from warnings import warn as _warn
import numpy as _np
import pandas as _pd

def get_signals(filename):
    """Get a list of signals in a signals file.

    :param filename: name of signals file with RxBasebandSignalNtfs
    :returns: table of signals
    """
    p = _re.compile(r'(\d+)\|RxBasebandSignalNtf:INFORM.* \((\d+) (baseband )?samples\).*')
    data = []
    lno = 0
    for s in open(filename, 'r'):
        lno += 1
        m = p.match(s)
        if m:
            t = int(m.group(1))
            m1 = _re.search(r' rxTime:(\d+) ', s)
            rxtime = int(m1.group(1)) if m1 else t
            m1 = _re.search(r' adc:(\d+) ', s)
            adc = int(m1.group(1)) if m1 else 1
            m1 = _re.search(r' channels:(\d+) ', s)
            ch = int(m1.group(1)) if m1 else 1
            m1 = _re.search(r' fc:(\d+) ', s)
            fc = int(m1.group(1)) if m1 else 0
            m1 = _re.search(r' fs:(\d+) ', s)
            fs = int(m1.group(1)) if m1 else 0
            m1 = _re.search(r' baseband samples', s)
            bb = True if m1 else False
            m1 = _re.search(r' preamble:(\d+) ', s)
            preamble = int(m1.group(1)) if m1 else 0
            m1 = _re.search(r' rssi:(\-?\d+\.?\d*) ', s)
            rssi = float(m1.group(1)) if m1 else _np.nan
            data.append([t, rxtime, adc, ch, fc, fs, bb, int(m.group(2)), preamble, rssi, filename, lno])
    if len(data) == 0:
        return None
    return _pd.DataFrame(data, columns=['time', 'rxtime', 'adc', 'channels', 'fc', 'fs', 'baseband', 'len', 'preamble', 'rssi', 'filename', 'lno'])

def get_signal(signals, n, order='F'):
    """Gets the specified signal from the list of signals.

    :param signals: table of signals returned by get_signals()
    :param n: signal index or signal selector
    :param order: ordering for multi-channel signals ('C' or 'F', see numpy.reshape())
    :returns: signal array

    >>> import arlpy.unet
    >>> s = arlpy.unet.get_signals('signals-0.txt')
    >>> x = arlpy.unet.get_signal(s, 2)
    >>> y = arlpy.unet.get_signal(s, s.rxtime == 123374675)
    """
    if type(n) == _pd.core.series.Series:
        n = signals[n].index
        if len(n) == 0:
            return None
        if len(n) > 1:
            _warn('Multiple signals found, returning first match')
        n = n.values[0]
    desired_lno = signals.lno[n] + 1
    lno = 0
    for s in open(signals.filename[n], 'r'):
        lno += 1
        if lno == desired_lno:
            x = _b64.standard_b64decode(s)
            if signals.baseband[n]:
                x = _np.array(_struct.unpack('>{0}f'.format(len(x)//4), x), dtype=_np.complex)
                x = x[0::2] + 1j*x[1::2]
            else:
                x = _np.array(_struct.unpack('>{0}f'.format(len(x)//4), x), dtype=_np.float)
            ch = signals.channels[n]
            if x.size != signals.len[n]*ch:
                _warn('Incorrect signal length')
            if ch > 1:
                if order == 'C':
                    x = _np.reshape(x, (x.size//ch, ch), order='C')
                else:
                    if order != 'F':
                        _warn('Unknown ordering: '+order+', assuming F')
                    x = _np.reshape(x, (ch, x.size//ch), order='F')
            return x

def read_signals(filename, callback, filter=None, order='F'):
    """Read a signals file and call callback for each signal.

    The callback function is called for each signal with a dictionary containing
    header information and the extracted signal.

    If a filter function is specified, it is called for each signal header.
    The function should return True if the signal should be extracted, False
    otherwise.

    :param filename: name of signals file with RxBasebandSignalNtfs
    :param callback: callback to call with each signal
    :param filter: callback to decide if a signal is extracted, or None
    :param order: ordering for multi-channel signals ('C' or 'F', see numpy.reshape())

    >>> import arlpy.unet
    >>> arlpy.unet.read_signals('signals-0.txt', lambda hdr, x: print(hdr, x.shape))
    >>> arlpy.unet.read_signals('signals-0.txt', lambda hdr, x: print(hdr, x.shape), lambda hdr: hdr['fc']==0)
    """
    p = _re.compile(r'(\d+)\|RxBasebandSignalNtf:INFORM .* \((\d+) (baseband )?samples\)')
    lno = 0
    accept = False
    for s in open(filename, 'r'):
        lno += 1
        m = p.match(s)
        if m:
            t = int(m.group(1))
            m1 = _re.search(r' rxTime:(\d+) ', s)
            rxtime = int(m1.group(1)) if m1 else t
            m1 = _re.search(r' adc:(\d+) ', s)
            adc = int(m1.group(1)) if m1 else 1
            m1 = _re.search(r' channels:(\d+) ', s)
            ch = int(m1.group(1)) if m1 else 1
            m1 = _re.search(r' fc:(\d+) ', s)
            fc = int(m1.group(1)) if m1 else 0
            m1 = _re.search(r' fs:(\d+) ', s)
            fs = int(m1.group(1)) if m1 else 0
            m1 = _re.search(r' baseband samples', s)
            bb = True if m1 else False
            m1 = _re.search(r' preamble:(\d+) ', s)
            preamble = int(m1.group(1)) if m1 else 0
            m1 = _re.search(r' rssi:(\-?\d+\.?\d*) ', s)
            rssi = float(m1.group(1)) if m1 else _np.nan
            hdr = { 'time': t, 'rxtime': rxtime, 'adc': adc, 'channels': ch, 'fc': fc, 'fs': fs,
                    'baseband': bb, 'len': int(m.group(2)), 'preamble': preamble, 'rssi': rssi,
                    'filename': filename, 'lno': lno }
            if filter:
                accept = filter(hdr)
            else:
                accept = True
        elif accept:
            accept = False
            try:
                x = _b64.standard_b64decode(s)
                if bb:
                    x = _np.array(_struct.unpack('>{0}f'.format(len(x)//4), x), dtype=_np.complex)
                    x = x[0::2] + 1j*x[1::2]
                else:
                    x = _np.array(_struct.unpack('>{0}f'.format(len(x)//4), x), dtype=_np.float)
                if x.size != hdr['len']*ch:
                    _warn('Incorrect signal length: '+filename+':'+str(lno))
                if ch > 1:
                    if order == 'C':
                        x = _np.reshape(x, (x.size//ch, ch), order='C')
                    else:
                        if order != 'F':
                            _warn('Unknown ordering: '+order+', assuming F')
                        x = _np.reshape(x, (ch, x.size//ch), order='F')
                callback(hdr, x)
            except Exception as ex:
                _warn('Bad signal: '+filename+':'+str(lno)+': '+str(ex))
