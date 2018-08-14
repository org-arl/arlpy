"""ROMANIS support toolbox."""

import numpy as _np
import os as _os
import warnings as _warn

_fs = 196000
_channels = 508
_framelen = 132
_datatype = _np.int16

def _close_all_files(fid):
    """Close the opened files.
    
    :param fid: list of file objects
    """
    for f in fid:
        f.close()

def _fread(fid, dt=_datatype, length=1, skip=0):
    """Read datafile of a channel with skip.
    
    :param fid: list of file objects
    :param dt: data type
    :param length: selected data length
    :param skip: data length to be skipped
    :returns: selected data (one channel)
    """
    data = _np.zeros((length))
    k = 0
    while k < length:
        data[k] = _np.fromfile(fid, dt, 1)
        fid.seek(skip, 1)
        k += 1
    return data

def get_sampling_rate(dirname=None):
    """Get the sampling rate in Hz.

    :param dirname: directory of the datafile
    :returns: sampling rate in Hz
    """
    return _fs

def get_channels(dirname=None):
    """Get the number of available data channels.

    :param dirname: directory of the datafile
    :returns: number of channels
    """
    return _channels

def get_data_length(dirname):
    """Get the length of the datafile in seconds.
    
    :param dirname: directory of the datafile
    :returns: data length in seconds 
    """
    files = 8  
    fid = [None]*files
    file_size = _np.zeros((files))
    for j in range(files):
        filename = _os.path.join(dirname, 'DAQ_{}'.format(j+1))
        fid[j] = open(filename, 'r')
        fid[j].seek(0, 2)
        file_size[j] = fid[j].tell()  
    datalen = min(file_size)
    datalen = _np.floor(datalen/_framelen)
    data = datalen/_fs
    _close_all_files(fid)
    return data

def get_data(dirname, start=0, length=None, calib=None, sensor=None):
    """Load selected data from ROMANIS recording.
    
    :param dirname: directory of the datafile
    :param start: start time (seconds, default is 0)
    :param length: data length (seconds, None means read all)
    :param calib: directory of the calibration file (None means no calibration)
    :param sensor: sensor number to load (None means load all, 0-507)
    :param returns: 2-D array of the selected data 
    """
    files = 8
    sensors_per_file = 64
    bytes_per_sample = 2
    header_samples = 2
    
    fid = [None]*files
    file_size = _np.zeros((files, 1))
    for j in range(files):
        filename = _os.path.join(dirname, 'DAQ_{}'.format(j+1))
        fid[j] = open(filename, 'rb')
        fid[j].seek(0, 2)
        file_size[j] = fid[j].tell()  
        
    datalen = min(file_size)
    if any(file_size != datalen):
        _warn.warn('Files are of unequal size')
    if (datalen%_framelen) is not 0:
        _warn.warn('File size is not an integral multiple of frame size')
    datalen = _np.floor(datalen/_framelen)
    start = int(round(start*_fs))      
    if ((length == None) and (calib == None) and (sensor == None)) or (length == None):
        length = datalen - start
    else:
        length = int(round(length*_fs))
    if (start < 0) or (start >= datalen):
        _close_all_files(fid)
        raise ValueError('Start out of range')
    if start+length > datalen:
        length = datalen - start
        _warn.warn('Insufficient data, returning less data than requested')
    if sensor is None:
        channels = _channels
    elif (sensor >= 0) and (sensor < _channels):
        channels = 1
    else:
        raise ValueError('Sensor number out of range')

    data = _np.zeros((length, channels), dtype=_datatype)
    for j in range(files):
        s0 = j*sensors_per_file
        s1 = min(s0+sensors_per_file, _channels)
        if channels == 1:
            if (sensor >= s0) and (sensor < s1): 
                skip = _framelen-bytes_per_sample
                fid[j].seek(start*_framelen+(sensor-s0+header_samples)*bytes_per_sample, 0)
                data[:, 0] = _fread(fid[j], _datatype, int(length), skip) 
                break
        else:
            n = s1-s0
            fid[j].seek(start*_framelen, 0)
            x = _np.fromfile(fid[j], _datatype, int(length*_framelen/bytes_per_sample))
            x = x.reshape((length, int(_framelen/bytes_per_sample)))
            data[:, range(s0, s1)] = x[:, header_samples+_np.arange(n)]
    _close_all_files(fid)

    data = data.astype(_np.float64)
    if calib is not None:
        calib_data = _np.loadtxt(calib)
        if channels == 1:
            data = (data-calib_data[sensor, 1])*calib_data[sensor, 0]
        else:
            for j in range(_channels):
                data[:,j] = (data[:, j]-calib_data[j, 1])*calib_data[j, 0]
    return data