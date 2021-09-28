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
    c = 1448.96 + 4.591*temperature - 5.304e-2*temperature**2 + 2.374e-4*temperature**3
    c += 1.340*(salinity-35) + 1.630e-2*depth + 1.675e-7*depth**2
    c += -1.025e-2*temperature*(salinity-35) - 7.139e-13*temperature*depth**3
    return c

def absorption(frequency, distance=1000, temperature=27, salinity=35, depth=10, pH=8.1):
    """Get the acoustic absorption in water.

    Computes acoustic absorption in water using Francois-Garrison model.

    :param frequency: frequency in Hz
    :param distance: distance in m
    :param temperature: temperature in deg C
    :param salinity: salinity in ppt
    :param depth: depth in m
    :param pH: pH of water
    :returns: absorption as a linear multiplier

    >>> import arlpy
    >>> arlpy.uwa.absorption(50000)
    0.2914
    >>> arlpy.utils.mag2db(arlpy.uwa.absorption(50000))
    -10.71
    >>> arlpy.utils.mag2db(arlpy.uwa.absorption(50000, distance=3000))
    -32.13
    """
    f = frequency/1000.0
    d = distance/1000.0
    c = 1412.0 + 3.21*temperature + 1.19*salinity + 0.0167*depth
    A1 = 8.86/c * 10**(0.78*pH-5)
    P1 = 1.0
    f1 = 2.8*_np.sqrt(salinity/35) * 10**(4-1245/(temperature+273))
    A2 = 21.44*salinity/c*(1+0.025*temperature)
    P2 = 1.0 - 1.37e-4*depth + 6.2e-9*depth*depth
    f2 = 8.17 * 10**(8-1990/(temperature+273)) / (1+0.0018*(salinity-35))
    P3 = 1.0 - 3.83e-5*depth + 4.9e-10*depth*depth
    if temperature < 20:
        A3 = 4.937e-4 - 2.59e-5*temperature + 9.11e-7*temperature*temperature - 1.5e-8*temperature*temperature*temperature
    else:
        A3 = 3.964e-4 - 1.146e-5*temperature + 1.45e-7*temperature*temperature - 6.5e-10*temperature*temperature*temperature
    a = A1*P1*f1*f*f/(f1*f1+f*f) + A2*P2*f2*f*f/(f2*f2+f*f) + A3*P3*f*f
    return 10**(-a*d/20.0)

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

    Computes sea water density using Fofonoff (1985 - IES 80).

    :param temperature: temperature in deg C
    :param salinity: salinity in ppt
    :returns: density in kg/m^3

    >>> import arlpy
    >>> arlpy.uwa.density()
    1022.7
    """
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
    return (1+speed/float(c))*frequency

def bubble_resonance(radius, depth=0.0, gamma = 1.4, p0 = 1.013e5, rho_water = 1022.476):
    """Compute resonance frequency of a freely oscillating has bubble in water,
    using implementation based on Medwin & Clay (1998). This ignores surface-tension,
    thermal, viscous and acoustic damping effects, and the pressure-volume relationship
    is taken to be adiabatic. Parameters:

    :radius: bubble `radius` in meters
    :depth: depth of bubble in water in meters
    :gamma: gas ratio of specific heats. Default 1.4 for air
    :p0: atmospheric pressure. Default 1.013e5 Pa
    :rho_water: Density of water. Default 1022.476 kg/m³

    >>> import arlpy
    >>> arlpy.uwa.bubble_resonance(100e-6)
    32465.56
    """
    g = 9.80665 #acceleration due to gravity
    p_air = p0 + rho_water*g*depth
    return 1/(2*_np.pi*radius)*_np.sqrt(3*gamma*p_air/rho_water)

def bubble_surface_loss(windspeed, frequency, angle):
    """Get the surface loss due to bubbles.

    The surface loss is computed based on APL model (1994).

    :param windspeed: windspeed in m/s (measured 10 m above the sea surface)
    :param frequency: frequency in Hz
    :param angle: incidence angle in radians
    :returns: absorption as a linear multiplier

    >>> import numpy
    >>> import arlpy
    >>> arlpy.utils.mag2db(uwa.bubble_surface_loss(3,10000,0))
    -1.44
    >>> arlpy.utils.mag2db(uwa.bubble_surface_loss(10,10000,0))
    -117.6
    """
    beta = _np.pi/2-angle
    f = frequency/1000.0
    if windspeed >= 6:
        a = 1.26e-3/_np.sin(beta) * windspeed**1.57 * f**0.85
    else:
        a = 1.26e-3/_np.sin(beta) * 6**1.57 * f**0.85 * _np.exp(1.2*(windspeed-6))
    return 10**(-a/20.0)

def bubble_soundspeed(void_fraction, c=soundspeed(), c_gas=340, relative_density=1000):
    """Get the speed of sound in a 2-phase bubbly water.

    The sound speed is computed based on Wood (1964) or Buckingham (1997).

    :param void_fraction: void fraction
    :param c: speed of sound in water in m/s
    :param c_gas: speed of sound in gas in m/s
    :param relative_density: ratio of density of water to gas
    :returns: sound speed in m/s

    >>> import arlpy
    >>> arlpy.uwa.bubble_soundspeed(1e-5)
    1402.133
    """
    m = _np.sqrt(relative_density)
    return 1/(1/c*_np.sqrt((void_fraction*(c/c_gas)**2*m+(1-void_fraction)/m)*(void_fraction/m+(1-void_fraction)*m)))

def pressure(x, sensitivity, gain, volt_params=None):
    """Convert the real signal x to an acoustic pressure signal in micropascal.

    :param x: real signal in voltage or bit depth (number of bits)
    :param sensitivity: receiving sensitivity in dB re 1V per micropascal
    :param gain: preamplifier gain in dB
    :param volt_params: (nbits, v_ref) is used to convert the number of bits
        to voltage where nbits is the number of bits of each sample and v_ref
        is the reference voltage, default to None
    :returns: acoustic pressure signal in micropascal

    If `volt_params` is provided, the sample unit of x is in number of bits,
    else is in voltage.

    >>> import arlpy
    >>> nbits = 16
    >>> V_ref = 1.0
    >>> x_volt = V_ref*signal.cw(64, 1, 512)
    >>> x_bit = x_volt*(2**(nbits-1))
    >>> sensitivity = 0
    >>> gain = 0
    >>> p1 = arlpy.uwa.pressure(x_volt, sensitivity, gain)
    >>> p2 = arlpy.uwa.pressure(x_bit, sensitivity, gain, volt_params=(nbits, V_ref))
    """
    nu = 10**(sensitivity/20)
    G = 10**(gain/20)
    if volt_params is not None:
        nbits, v_ref = volt_params
        x = x*v_ref/(2**(nbits-1))
    return x/(nu*G)

def spl(x, ref=1):
    """Get Sound Pressure Level (SPL) of the acoustic pressure signal x.

    :param x: acoustic pressure signal in micropascal
    :param ref: reference acoustic pressure in micropascal, default to 1
    :returns: average SPL in dB re micropascal

    In water, the common reference is 1 micropascal. In air, the common
    reference is 20 micropascal.

    >>> import arlpy
    >>> p = signal.cw(64, 1, 512)
    >>> arlpy.uwa.spl(p)
    -3.0103
    """
    rmsx = _np.sqrt(_np.mean(_np.power(_np.abs(x), 2)))
    return 20*_np.log10(rmsx/ref)
