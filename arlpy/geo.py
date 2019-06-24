##############################################################################
#
# Copyright (c) 2016, Mandar Chitre
#
# This file is part of arlpy which is released under Simplified BSD License.
# See file LICENSE or go to http://www.opensource.org/licenses/BSD-3-Clause
# for full license details.
#
##############################################################################

"""Geographical coordinates toolbox.

This toolbox provides functions to work with different geographical coordinate systems.
Latitude/longitude position is represented by parameter `latlong` with one of these
formats:

   * `(latitude degrees, longitude degrees)`
   * `(latitude degrees, longitude degrees, altitude)`
   * `(latitude degrees, minutes, longitude degrees, minutes)`
   * `(latitude degrees, minutes, longitude degrees, minutes, altitude)`
   * `(latitude degrees, minutes, seconds, longitude degrees, minutes, seconds)`
   * `(latitude degrees, minutes, seconds, longitude degrees, minutes, seconds, altitude)`

The `altitude` is always specified in meters, with negative values being depth below water
surface.

`pos` represents position in a local coordinate system `(easting, northing)` or
`(easting, northing, altitude)` specified by a UTM zone.
"""

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
    latlong = list(map(float, latlong))
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

def pos(latlong, zonenum=None, origin=None):
    """Convert latitude/longitude to local coordinate system position.

    If an origin is specified, the local coordinate system is set up with that origin
    and East-North axis. If no origin is specified, the UTM local coordinate system
    is used. A specific UTM zone can be forced by specifying `zonenum`, if desired.
    """
    latlong = _normalize(latlong)
    pos = _utm.from_latlon(latlong[0], latlong[1], zonenum)
    opos = (0,0)
    if origin is not None:
        origin = _normalize(origin)
        opos = _utm.from_latlon(origin[0], origin[1], zonenum)
    return (pos[0]-opos[0], pos[1]-opos[1], latlong[2])

def zone(latlong):
    """Convert latitude/longitude to UTM zone."""
    latlong = _normalize(latlong)
    pos = _utm.from_latlon(latlong[0], latlong[1])
    return pos[2:4]

def latlong(pos, zone=None, origin=None):
    """Convert local coordinate system position to latitude/longitude.

    To convert a UTM position into a global latitude/longitude, the local coordinate
    system has to be specified in terms of a UTM zone 2-tuple, e.g. ``(32, 'U')``.
    Alternatively a local coordinate system can be specified in terms of an origin
    latitude/longitude.
    """
    opos = (0,0)
    if origin is not None:
        if zone is not None:
            raise ValueError('zone and origin cannot be concurrently specified')
        origin = _normalize(origin)
        opos = _utm.from_latlon(origin[0], origin[1])
        zone = (opos[2], opos[3])
    geo = _utm.to_latlon(pos[0]+opos[0], pos[1]+opos[1], zone[0], zone[1])
    return (geo[0], geo[1], 0.0 if len(pos) < 3 else pos[2])

def d(latlong):
    """Convert latitude/longitude to (latitude degrees, longitude degrees) format."""
    return _normalize(latlong)[:-1]

def dm(latlong):
    """Convert latitude/longitude to (latitude degrees, minutes, longitude degrees, minutes) format."""
    latlon = _normalize(latlong)
    return (_floor(latlon[0]), _frac(latlon[0])*60.0, _floor(latlon[1]), _frac(latlon[1])*60.0)

def dms(latlong):
    """Convert latitude/longitude to (latitude degrees, minutes, seconds, longitude degrees, minutes, seconds) format."""
    latlon = _normalize(latlong)
    m1 = _frac(latlon[0])*60.0
    m2 = _frac(latlon[1])*60.0
    return (_floor(latlon[0]), _floor(m1), _frac(m1)*60.0, _floor(latlon[1]), _floor(m2), _frac(m2)*60.0)

def dz(latlong):
    """Convert latitude/longitude to (latitude degrees, longitude degrees, altitude) format."""
    return _normalize(latlong)

def dmz(latlong):
    """Convert latitude/longitude to (latitude degrees, minutes, longitude degrees, minutes, altitude) format."""
    latlon = _normalize(latlong)
    return (_floor(latlon[0]), _frac(latlon[0])*60.0, _floor(latlon[1]), _frac(latlon[1])*60.0, latlon[2])

def dmsz(latlong):
    """Convert latitude/longitude to (latitude degrees, minutes, seconds, longitude degrees, minutes, seconds, altitude) format."""
    latlon = _normalize(latlong)
    m1 = _frac(latlon[0])*60.0
    m2 = _frac(latlon[1])*60.0
    return (_floor(latlon[0]), _floor(m1), _frac(m1)*60.0, _floor(latlon[1]), _floor(m2), _frac(m2)*60.0, latlon[2])

def distance(pos1, pos2):
    """Compute distance between two UTM positions."""
    return _np.linalg.norm(_np.asarray(pos1)-_np.asarray(pos2))

def str(latlong):
    """Convert latitude/longitude information in various formats into a pretty printable Unicode string."""
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
