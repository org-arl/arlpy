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

import os as _os
import sys as _sys
import uuid as _uuid
import numpy as _np

_notebook = False

try:
    get_ipython                     # check if we are using IPython
    _os.environ['JPY_PARENT_PID']   # and Jupyter
    import IPython.display as _ipyd
    _ipyd.ProgressBar               # and IPython >= 6.2.1
    _notebook = True
except:
    pass                            # not in Jupyter, skip notebook initialization

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

def rotation_matrix(alpha, beta, gamma):
    """Generates a 3D rotation matrix.

    :param alpha: rotation angle around x-axis
    :param beta: rotation angle around y-axis
    :param gamma: rotation angle around z-axis

    Rotation is applied around x, y and z axis in that order.
    """
    R = _np.eye(3)
    if alpha != 0:
        R = _np.dot(_np.array([[1.,             0.,              0.],
                               [0., _np.cos(alpha), -_np.sin(alpha)],
                               [0., _np.sin(alpha),  _np.cos(alpha)]]), R)
    if beta != 0:
        R = _np.dot(_np.array([[ _np.cos(beta), 0., _np.sin(beta)],
                               [            0., 1.,            0.],
                               [-_np.sin(beta), 0., _np.cos(beta)]]), R)

    if gamma != 0:
        R = _np.dot(_np.array([[_np.cos(gamma), -_np.sin(gamma), 0.],
                               [_np.sin(gamma),  _np.cos(gamma), 0.],
                               [            0.,              0., 1.]]), R)
    return R

def broadcastable_to(x, shape, axis=None):
    """Convert 1D array to be broadcastable along specified axis.

    :param x: array to broadcast
    :param shape: shape to broadcast to
    :param axis: axis to broadcast along

    Reshapes the array to the minimum dimensions such that it can be
    broadcasted to the given shape along the specified axis. If an axis
    is not specified, the first axis that matches the size of x is used.

    >>> import arlpy
    >>> import numpy as np
    >>> arlpy.utils.broadcastable_to(np.array([1,2,3]), (5,3,2), 1)
    array([[1],
           [2],
           [3]])
    """
    x = _np.asarray(x)
    if x.ndim != 1:
        raise ValueError('x should be a 1D array')
    if axis is None:
        axis = _np.argmax(len(x) == _np.array(shape))
        if len(x) != shape[axis]:
            raise ValueError('No axis matches length of x')
    if axis < 0:
        axis += len(shape)
    if axis < 0 or axis >= len(shape):
        raise ValueError('Bad axis specification')
    if len(x) != shape[axis]:
        raise ValueError('x should be (%d,) array, but is %s'%(shape[axis],str(x.shape)))
    for j in range(len(shape)-axis-1):
        x = x[:,_np.newaxis]
    return x

def progress(n, width=50):
    """Display progress bar for long running operations.

    :param n: total number of steps to completion
    :param width: width of the progress bar (only for the text version)

    >>> import arlpy
    >>> progress = arlpy.utils.progress(100)
    >>> for j in range(100):
            next(progress)
    """
    if _notebook:
        import IPython.display as _ipyd
        p = _ipyd.ProgressBar(total=n)
        did = str(_uuid.uuid4())
        _ipyd.display(p, display_id=did)
        for j in range(1, n):
            p.progress = j
            _ipyd.update_display(p, display_id=did)
            yield j
        _ipyd.update_display(_ipyd.HTML(''), display_id=did)
        yield None
    else:
        _sys.stdout.write('%s|\n' % ('-'*width))
        _sys.stdout.flush()
        c = 0
        for j in range(n):
            c1 = int(width*(j+1)/n)
            if c1 > c:
                _sys.stdout.write('>'*(c1-c))
                c = c1
                if c == width:
                    _sys.stdout.write('\n')
                _sys.stdout.flush()
            yield j
