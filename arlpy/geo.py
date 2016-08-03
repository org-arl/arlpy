"""Geographical coordinates toolbox."""

from math import fabs as _fabs
import numpy as _np
import utm as _utm

def _floor(x):
    return float(int(x))

def _frac(x):
    return _fabs(x-_floor(x))

def _ns(x):
    return 'S' if x < 0 else 'N'

def _ew(x):
    return 'W' if x < 0 else 'E'

def _normalize(latlong):
    latlong = map(float, latlong)
    if len(latlong) == 2:
        return (latlong[0], latlong[1], 0.0)
    elif len(latlong) == 3:
        return tuple(latlong)
    elif len(latlong) == 4:
        return (latlong[0]+latlong[1]/60.0, latlong[2]+latlong[3]/60.0, 0.0)
    elif len(latlong) == 5:
        return (latlong[0]+latlong[1]/60.0, latlong[2]+latlong[3]/60.0, latlong[4])
    elif len(latlong) == 6:
        return (latlong[0]+latlong[1]/60.0+latlong[2]/3600.0, latlong[3]+latlong[4]/60.0+latlong[5]/3600.0, 0.0)
    elif len(latlong) == 7:
        return (latlong[0]+latlong[1]/60.0+latlong[2]/3600.0, latlong[3]+latlong[4]/60.0+latlong[5]/3600.0, latlong[6])
    else:
        raise ValueError('Incorrect format for latitude/longitude data')

def pos(latlong, zonenum=None):
    """Convert (latitude, longitude, altitude) to UTM (easting, northing, altitude)."""
    latlong = _normalize(latlong)
    pos = _utm.from_latlon(latlong[0], latlong[1], zonenum)
    return (pos[0], pos[1], latlong[2])

def zone(latlong):
    """Convert (latitude, longitude, altitude) to UTM zone."""
    latlong = _normalize(latlong)
    pos = _utm.from_latlon(latlong[0], latlong[1])
    return pos[2:4]

def latlong(pos, zone):
    """Convert UTM (easting, northing, altitude) to (latitude, longitude, altitude)."""
    geo = _utm.to_latlon(pos[0], pos[1], zone[0], zone[1])
    return (geo[0], geo[1], 0.0 if len(pos) < 3 else pos[2])

def d(latlong):
    """Convert (latitude, longitude) to decimal degrees format."""
    return _normalize(latlong)[:-1]

def dm(latlong):
    """Convert (latitude, longitude) to degrees/minutes format."""
    latlon = _normalize(latlong)
    return (_floor(latlon[0]), _frac(latlon[0])*60.0, _floor(latlon[1]), _frac(latlon[1])*60.0)

def dms(latlong):
    """Convert (latitude, longitude) to degrees/minutes/seconds format."""
    latlon = _normalize(latlong)
    m1 = _frac(latlon[0])*60.0
    m2 = _frac(latlon[1])*60.0
    return (_floor(latlon[0]), _floor(m1), _frac(m1)*60.0, _floor(latlon[1]), _floor(m2), _frac(m2)*60.0)

def dz(latlong):
    """Convert (latitude, longitude, altitude) to decimal degrees format."""
    return _normalize(latlong)

def dmz(latlong):
    """Convert (latitude, longitude, altitude) to degrees, minutes format."""
    latlon = _normalize(latlong)
    return (_floor(latlon[0]), _frac(latlon[0])*60.0, _floor(latlon[1]), _frac(latlon[1])*60.0, latlon[2])

def dmsz(latlong):
    """Convert (latitude, longitude, altitude) to degrees, minutes, seconds format."""
    latlon = _normalize(latlong)
    m1 = _frac(latlon[0])*60.0
    m2 = _frac(latlon[1])*60.0
    return (_floor(latlon[0]), _floor(m1), _frac(m1)*60.0, _floor(latlon[1]), _floor(m2), _frac(m2)*60.0, latlon[2])

def distance(pos1, pos2):
    """Compute distance between two points specified as (easting, northing, altitude)."""
    return _np.linalg.norm(_np.asarray(pos1)-_np.asarray(pos2))

def str(latlong):
    """Convert (latitude, longitude, altitude) information in various formats into a pretty printable string."""
    if len(latlong) == 2:
        return u'{0:f}\N{DEGREE SIGN}{1:s}, {2:f}\N{DEGREE SIGN}{3:s}'.format(_fabs(latlong[0]), _ns(latlong[0]), _fabs(latlong[1]), _ew(latlong[1]))
    elif len(latlong) == 3:
        return u'{0:f}\N{DEGREE SIGN}{1:s}, {2:f}\N{DEGREE SIGN}{3:s}, {4:.3f}m'.format(_fabs(latlong[0]), _ns(latlong[0]), _fabs(latlong[1]), _ew(latlong[1]), latlong[2])
    elif len(latlong) == 4:
        return u'{0:.0f}\N{DEGREE SIGN}{1:.4f}\'{2:s}, {3:.0f}\N{DEGREE SIGN}{4:.4f}\'{5:s}'.format(_fabs(latlong[0]), latlong[1], _ns(latlong[0]), _fabs(latlong[2]), latlong[3], _ew(latlong[2]))
    elif len(latlong) == 5:
        return u'{0:.0f}\N{DEGREE SIGN}{1:.4f}\'{2:s}, {3:.0f}\N{DEGREE SIGN}{4:.4f}\'{5:s}, {6:.3f}m'.format(_fabs(latlong[0]), latlong[1], _ns(latlong[0]), _fabs(latlong[2]), latlong[3], _ew(latlong[2]), latlong[4])
    elif len(latlong) == 6:
        return u'{0:.0f}\N{DEGREE SIGN}{1:.0f}\'{2:.2f}"{3:s}, {4:.0f}\N{DEGREE SIGN}{5:.0f}\'{6:.2f}"{7:s}'.format(_fabs(latlong[0]), latlong[1], latlong[2], _ns(latlong[0]), _fabs(latlong[3]), latlong[4], latlong[5], _ew(latlong[3]))
    elif len(latlong) == 7:
        return u'{0:.0f}\N{DEGREE SIGN}{1:.0f}\'{2:.2f}"{3:s}, {4:.0f}\N{DEGREE SIGN}{5:.0f}\'{6:.2f}"{7:s}, {8:.3f}m'.format(_fabs(latlong[0]), latlong[1], latlong[2], _ns(latlong[0]), _fabs(latlong[3]), latlong[4], latlong[5], _ew(latlong[3]), latlong[6])
    else:
        raise ValueError('Incorrect format for latitude/longitude data')
