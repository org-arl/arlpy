##############################################################################
#
# Copyright (c) 2016, Mandar Chitre
#
# This file is part of arlpy which is released under Simplified BSD License.
# See file LICENSE or go to http://www.opensource.org/licenses/BSD-3-Clause
# for full license details.
#
##############################################################################

"""Stable distribution toolbox."""

import numpy as _np
import scipy.stats as _stats

# lookup table from McCulloch (1986)
_ena = _np.array([
        [   2.4388,    2.4388,    2.4388,    2.4388,    2.4388 ],
        [   2.5120,    2.5117,    2.5125,    2.5129,    2.5148 ],
        [   2.6080,    2.6093,    2.6101,    2.6131,    2.6174 ],
        [   2.7369,    2.7376,    2.7387,    2.7420,    2.7464 ],
        [   2.9115,    2.9090,    2.9037,    2.8998,    2.9016 ],
        [   3.1480,    3.1363,    3.1119,    3.0919,    3.0888 ],
        [   3.4635,    3.4361,    3.3778,    3.3306,    3.3161 ],
        [   3.8824,    3.8337,    3.7199,    3.6257,    3.5997 ],
        [   4.4468,    4.3651,    4.1713,    4.0052,    3.9635 ],
        [   5.2172,    5.0840,    4.7778,    4.5122,    4.4506 ],
        [   6.3140,    6.0978,    5.6241,    5.2195,    5.1256 ],
        [   7.9098,    7.5900,    6.8606,    6.2598,    6.1239 ],
        [  10.4480,    9.9336,    8.7790,    7.9005,    7.6874 ],
        [  14.8378,   13.9540,   12.0419,   10.7219,   10.3704 ],
        [  23.4831,   21.7682,   18.3320,   16.2163,   15.5841 ],
        [  44.2813,   40.1367,   33.0018,   29.1399,   27.7822 ]
])

def sstabfit(x):
    """Fit a symmetric alpha stable distribution to data.

    :param x: data
    :returns: (alpha, c, delta)

    alpha, c and delta are the characteristic exponent, scale parameter
    (dispersion^1/alpha) and location parameter respectively.

    alpha is computed based on McCulloch (1986) fractile.
    c is computed based on Fama & Roll (1971) fractile.
    delta is the 50% trimmed mean of the sample.
    """
    delta = _stats.trim_mean(x, 0.25)
    p = _np.percentile(x, [5, 25, 28, 72, 75, 95])
    c = (p[3]-p[2])/1.654
    an = (p[5]-p[0])/(p[4]-p[1])
    if an < 2.4388:
        alpha = 2
    else:
        alpha = 0
        j = _np.where(an <= _ena[:,0])[0]
        if len(j) > 0:
            j = j[0]
            t = (an-_ena[j-1,0])/(_ena[j,0]-_ena[j-1,0])
            alpha = (21-j-t)/10
    if alpha < 0.5:
        alpha = 0.5
    return (alpha, c, delta)

def rnd(alpha=1.5, beta=0, scale=1, loc=0.0, size=1):
    """Generate independent stable random numbers.

    :param alpha: characteristic exponent (0.1 to 2.0)
    :param beta: skew (-1 to +1)
    :param scale: scale parameter
    :param loc: location parameter (mean for alpha > 1, median/mode when beta=0)
    :param size: size of the random array to generate

    This implementation is based on the method in J.M. Chambers, C.L. Mallows
    and B.W. Stuck, "A Method for Simulating Stable Random Variables," JASA 71 (1976): 340-4.
    McCulloch's MATLAB implementation (1996) served as a reference in developing this code.
    """
    if alpha < 0.1 or alpha > 2:
        raise ValueError('alpha must be in the range 0.1 to 2')
    if _np.abs(beta) > 1:
        raise ValueError('beta must be in the range -1 to 1')
    phi = (_np.random.uniform(size=size) - 0.5) * _np.pi
    if alpha == 1 and beta == 0:
        return loc + scale*_np.tan(phi)
    w = -_np.log(_np.random.uniform(size=size))
    if alpha == 2:
        return loc + 2*scale*_np.sqrt(w)*_np.sin(phi)
    if beta == 0:
        return loc + scale * ((_np.cos((1-alpha)*phi) / w) ** (1.0/alpha - 1) * _np.sin(alpha * phi) / _np.cos(phi) ** (1.0/alpha))
    cosphi = _np.cos(phi)
    if _np.abs(alpha-1) > 1e-8:
        zeta = beta * _np.tan(_np.pi*alpha/2)
        aphi = alpha * phi
        a1phi = (1-alpha) * phi
        return loc + scale * (( (_np.sin(aphi)+zeta*_np.cos(aphi))/cosphi * ((_np.cos(a1phi)+zeta*_np.sin(a1phi))) / ((w*cosphi)**((1-alpha)/alpha)) ))
    bphi = _np.pi/2 + beta*phi
    x = 2/_np.pi * (bphi*_np.tan(phi) - beta*_np.log(_np.pi/2*w*cosphi/bphi))
    if alpha != 1:
        x += beta * _np.tan(_np.pi*alpha/2)
    return loc + scale*x
