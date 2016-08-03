"""Underwater acoustics toolbox."""

from math import sqrt as _sqrt

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
    if f > 3.0:
        # Marsh & Schulkin: JASA 1962
        A = 2.34e-6
        B = 3.38e-6
        P = (density(temperature, salinity) * depth + 1e5/9.8) * 1e-4
        fT = 21.9 * 10**(6-1520/(temperature+273))
        a = 8.68 * (salinity*A*fT*f*f/(fT*fT+f*f)+B*f*f/fT)*(1-6.54e-4*P)
    else:
        # Thorp & Browning: J. Sound Vib. 1973
        a = (0.11*f*f/(1+f*f)+44*f*f/(4100+f*f))*1e-3
    return 10**(-a*distance/20.0)

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
    return A + salinity * (B + C*_sqrt(salinity) + D*salinity)

def doppler(speed, frequency, soundspeed=soundspeed()):
    """Get the Doppler-shifted frequency given relative speed between transmitter and receiver."""
    return (1+speed/float(soundspeed))*frequency        # approximation holds when speed << c
