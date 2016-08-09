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
    """Get the speed of sound in water."""
    # Mackenzie: JASA 1981
    c = 1448.96 + 4.591*temperature - 5.304e-2*temperature**2 + 2.374e-4*temperature**3
    c += 1.340*(salinity-35) + 1.630e-2*depth + 1.675e-7*depth**2
    c += -1.025e-2*temperature*(salinity-35) - 7.139e-13*temperature*depth**3
    return c

def absorption(frequency, distance=1000, temperature=27, salinity=35, depth=10):
    """Get the acoustic absorption in water."""
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
    """Design a FIR filter with response based on acoustic absorption in water."""
    nyquist = fs/2.0
    f = _np.linspace(0, nyquist, num=nfreqs)
    g = absorption(f, distance, temperature, salinity, depth)
    return _sp.firwin2(ntaps, f, g, nyq=nyquist)

def density(temperature=27, salinity=35):
    """Get the density of sea water near the surface."""
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
    """Get the Rayleigh reflection coefficient for a given angle."""
    # Brekhovskikh & Lysanov
    n = float(c)/c1*(1+1j*alpha)
    m = float(rho1)/rho
    t1 = m*_np.cos(angle)
    t2 = _np.sqrt(n**2-_np.sin(angle)**2)
    V = (t1-t2)/(t1+t2)
    return V.real if V.imag == 0 else V

def doppler(speed, frequency, soundspeed=soundspeed()):
    """Get the Doppler-shifted frequency given relative speed between transmitter and receiver."""
    return (1+speed/float(soundspeed))*frequency        # approximation holds when speed << c
