##############################################################################
#
# Copyright (c) 2016, Mandar Chitre
#
# This file is part of arlpy which is released under Simplified BSD License.
# See file LICENSE or go to http://www.opensource.org/licenses/BSD-3-Clause
# for full license details.
#
##############################################################################

"""Import commonly used symbols into namespace.

While cluttering the global namespace is usually discouraged, having to reference
commonly needed math functions from the `numpy` or `arlpy` packages hampers readability.
The `arlpy.common` module is meant to provide a common set of math functions that
are imported into the global namespace. The use of this module is simply a matter of
personal preference on coding style.

Intended usage::

    from arlpy.common import *

This then allows all symbols from the :mod:`arlpy.utils` module, and the following common
symbols from the :mod:`numpy` module to be used without explicit qualification:

    * :func:`sqrt`
    * :func:`fabs`
    * :func:`log10`
    * :func:`log2`
    * :func:`log`
    * :func:`exp`
    * :func:`pi`
    * :func:`sin`
    * :func:`cos`
    * :func:`tan`
    * :func:`arcsin`
    * :func:`arccos`
    * :func:`arctan`
    * :func:`arctan2`

"""

from numpy import sqrt, fabs, log10, log2, log, exp, pi, sin, cos, tan, arcsin, arccos, arctan, arctan2
from .utils import *
