##############################################################################
#
# Copyright (c) 2016, Mandar Chitre
#
# This file is part of arlpy which is released under Simplified BSD License.
# See file LICENSE or go to http://www.opensource.org/licenses/BSD-3-Clause
# for full license details.
#
##############################################################################

"""Underwater acoustics toolbox."""

import numbers as _num
import numpy as _np
import scipy.signal as _sp

def soundspeed(temperature=27, salinity=35, depth=10):
    """Get the speed of sound in water.

    Uses Mackenzie (1981) to compute sound speed in water.

    :param temperature: temperature in deg C
    :param salinity: salinity in ppt
    :param depth: depth in m
    :returns: sound speed in m/s

    >>> import arlpy
    >>> arlpy.uwa.soundspeed()
    1539.1
    >>> arlpy.uwa.soundspeed(temperature=25, depth=20)
    1534.6
    """
    # Mackenzie: JASA 1981
    c = 1448.96 + 4.591*temperature - 5.304e-2*temperature**2 + 2.374e-4*temperature**3
    c += 1.340*(salinity-35) + 1.630e-2*depth + 1.675e-7*depth**2
    c += -1.025e-2*temperature*(salinity-35) - 7.139e-13*temperature*depth**3
    return c

def absorption(frequency, distance=1000, temperature=27, salinity=35, depth=10):
    """Get the acoustic absorption in water.

    Computes acoustic absorption in water using Marsh & Schulkin (1962) for
    frequency above 3 kHz, and Thorp & Browning (1973) for lower frequencies.

    :param frequency: frequency in Hz
    :param distance: distance in m
    :param temperature: temperature in deg C
    :param salinity: salinity in ppt
    :param depth: depth in m
    :returns: absorption as a linear multiplier

    >>> import arlpy
    >>> arlpy.uwa.absorption(50000)
    0.3964
    >>> arlpy.utils.mag2db(arlpy.uwa.absorption(50000))
    -8.04
    >>> arlpy.utils.mag2db(arlpy.uwa.absorption(50000, distance=3000))
    -24.11
    """
    f = frequency/1000.0
    # Marsh & Schulkin: JASA 1962
    A = 2.34e-6
    B = 3.38e-6
    P = (density(temperature, salinity) * depth + 1e5/9.8) * 1e-4
    fT = 21.9 * 10**(6-1520/(temperature+273))
    a1 = 8.68 * (salinity*A*fT*f*f/(fT*fT+f*f)+B*f*f/fT)*(1-6.54e-4*P)
    # Thorp & Browning: J. Sound Vib. 1973
    a2 = (0.11*f*f/(1+f*f)+44*f*f/(4100+f*f))*1e-3
    # a1 is valid for f > 3, otherwise a2 is valid
    if isinstance(a1, _num.Number):
        a = a1 if f > 3.0 else a2
    else:
        a = _np.where(f>3.0, a1, a2)
    return 10**(-a*distance/20.0)

def absorption_filter(fs, ntaps=31, nfreqs=64, distance=1000, temperature=27, salinity=35, depth=10):
    """Design a FIR filter with response based on acoustic absorption in water.

    :param fs: sampling frequency in Hz
    :param ntaps: number of FIR taps
    :param nfreqs: number of frequencies to use for modeling frequency response
    :param distance: distance in m
    :param temperature: temperature in deg C
    :param salinity: salinity in ppt
    :param depth: depth in m
    :returns: tap weights for a FIR filter that represents absorption at the given distance

    >>> import arlpy
    >>> import numpy as np
    >>> fs = 250000
    >>> b = arlpy.uwa.absorption_filter(fs, distance=500)
    >>> x = arlpy.signal.sweep(20000, 40000, 0.5, fs)
    >>> y = arlpy.signal.lfilter0(b, 1, x)
    >>> y /= 500**2      # apply spreading loss for 500m
    """
    nyquist = fs/2.0
    f = _np.linspace(0, nyquist, num=nfreqs)
    g = absorption(f, distance, temperature, salinity, depth)
    return _sp.firwin2(ntaps, f, g, nyq=nyquist)

def density(temperature=27, salinity=35):
    """Get the density of sea water near the surface.

    Computes sea water density using Fofonoff (1985).

    :param temperature: temperature in deg C
    :param salinity: salinity in ppt
    :returns: density in kg/m^3

    >>> import arlpy
    >>> arlpy.uwa.density()
    1022.7
    """
    # Fofonoff: JGR 1985 (IES 80)
    t = temperature
    A = 1.001685e-04 + t * (-1.120083e-06 + t * 6.536332e-09)
    A = 999.842594 + t * (6.793952e-02 + t * (-9.095290e-03 + t * A))
    B = 7.6438e-05 + t * (-8.2467e-07 + t * 5.3875e-09)
    B = 0.824493 + t * (-4.0899e-03 + t * B)
    C = -5.72466e-03 + t * (1.0227e-04 - t * 1.6546e-06)
    D = 4.8314e-04
    return A + salinity * (B + C*_np.sqrt(salinity) + D*salinity)

def reflection_coeff(angle, rho1, c1, alpha=0, rho=density(), c=soundspeed()):
    """Get the Rayleigh reflection coefficient for a given angle.

    :param angle: angle of incidence in radians
    :param rho1: density of second medium in kg/m^3
    :param c1: sound speed in second medium in m/s
    :param alpha: attenuation
    :param rho: density of water in kg/m^3
    :param c: sound speed in water in m/s
    :returns: reflection coefficient as a linear multiplier

    >>> from numpy import pi
    >>> import arlpy
    >>> arlpy.uwa.reflection_coeff(pi/4, 1200, 1600)
    0.1198
    >>> arlpy.uwa.reflection_coeff(0, 1200, 1600)
    0.0990
    >>> arlpy.utils.mag2db(arlpy.uwa.reflection_coeff(0, 1200, 1600))
    -20.1
    """
    # Brekhovskikh & Lysanov
    n = float(c)/c1*(1+1j*alpha)
    m = float(rho1)/rho
    t1 = m*_np.cos(angle)
    t2 = _np.sqrt(n**2-_np.sin(angle)**2)
    V = (t1-t2)/(t1+t2)
    return V.real if V.imag == 0 else V

def doppler(speed, frequency, c=soundspeed()):
    """Get the Doppler-shifted frequency given relative speed between transmitter and receiver.

    The Doppler approximation used is only valid when `speed` << `c`. This is usually the case
    for underwater vehicles.

    :param speed: relative speed between transmitter and receiver in m/s
    :param frequency: transmission frequency in Hz
    :param c: sound speed in m/s
    :returns: the Doppler shifted frequency as perceived by the receiver

    >>> import arlpy
    >>> arlpy.uwa.doppler(2, 50000)
    50064.97
    >>> arlpy.uwa.doppler(-1, 50000)
    49967.51
    """
    return (1+speed/float(c))*frequency        # approximation holds when speed << c
