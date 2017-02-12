##############################################################################
#
# Copyright (c) 2016, Mandar Chitre
#
# This file is part of arlpy which is released under Simplified BSD License.
# See file LICENSE or go to http://www.opensource.org/licenses/BSD-3-Clause
# for full license details.
#
##############################################################################

"""Common utility functions."""

import numpy as _np
import sys as _sys

def mag2db(x):
    """Convert magnitude quantity to dB."""
    return 20*_np.log10(x)

def pow2db(x):
    """Convert power quantity to dB."""
    return 10*_np.log10(x)

def db2mag(x):
    """Convert dB quantity to magnitude."""
    return _np.power(10, x/20.0)

def db2pow(x):
    """Convert dB quantity to power."""
    return _np.power(10, x/10.0)

def linspace2d(start0, stop0, num0, start1, stop1, num1):
    """Generate linearly spaced coordinates in 2D space.

    :param start0: first value on axis 0
    :param stop0: last value on axis 0
    :param num0: number of values on axis 0
    :param start1: first value on axis 1
    :param stop1: last value on axis 1
    :param num1: number of values on axis 1

    >>> from arlpy import bf
    >>> bf.linspace2d(0, 1, 2, 0, 1, 3)
    [[ 0. ,  0. ],
     [ 0. ,  0.5],
     [ 0. ,  1. ],
     [ 1. ,  0. ],
     [ 1. ,  0.5],
     [ 1. ,  1. ]]
    """
    x = _np.linspace(start0, stop0, num0, dtype=_np.float)
    y = _np.linspace(start1, stop1, num1, dtype=_np.float)
    return _np.array(_np.meshgrid(x, y)).T.reshape(-1, 2)

def linspace3d(start0, stop0, num0, start1, stop1, num1, start2, stop2, num2):
    """Generate linearly spaced coordinates in 2D space.

    :param start0: first value on axis 0
    :param stop0: last value on axis 0
    :param num0: number of values on axis 0
    :param start1: first value on axis 1
    :param stop1: last value on axis 1
    :param num1: number of values on axis 1
    :param start2: first value on axis 2
    :param stop2: last value on axis 2
    :param num2: number of values on axis 2

    >>> from arlpy import bf
    >>> bf.linspace3d(0, 1, 2, 0, 1, 3, 0, 0, 1)
    [[ 0. ,  0. , 0. ],
     [ 0. ,  0.5, 0. ],
     [ 0. ,  1. , 0. ],
     [ 1. ,  0. , 0. ],
     [ 1. ,  0.5, 0. ],
     [ 1. ,  1. , 0. ]]
    """
    x = _np.linspace(start0, stop0, num0, dtype=_np.float)
    y = _np.linspace(start1, stop1, num1, dtype=_np.float)
    z = _np.linspace(start2, stop2, num2, dtype=_np.float)
    return _np.array(_np.meshgrid(x, y, z)).T.reshape(-1, 3)

def progress(n, width=50):
    """Display progress bar for long running operations.

    :param n: total number of steps to completion
    :param width: width of the progress bar

    >>> import arlpy
    >>> progress = arlpy.utils.progress(100)
    >>> for j in range(100):
            next(progress)
    """
    _sys.stdout.write('%s|\n' % ('-'*width))
    _sys.stdout.flush()
    c = 0
    for j in xrange(n):
        c1 = width*(j+1)/n
        if c1 > c:
            _sys.stdout.write('>'*(c1-c))
            c = c1
            if c == width:
                _sys.stdout.write('\n')
            _sys.stdout.flush()
        yield j
