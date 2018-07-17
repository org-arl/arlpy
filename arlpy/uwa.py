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

import os as _os
import numbers as _num
import numpy as _np
import scipy.signal as _sp
from tempfile import mkstemp as _mkstemp

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
    -9.21
    >>> arlpy.utils.mag2db(arlpy.uwa.absorption(50000, distance=3000))
    -27.64
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

def pm_environment(**env):
    """Define an environment for an underwater acoustic propagation model.
    """
    e = {
        'frequency': 25000,
        'soundspeed': 1500,
        'bottom_soundspeed': 1600,
        'bottom_density': 1.6,
        'bottom_absorption': 0.1,
        'bottom_roughness': 0,
        'tx_depth': 5,
        'rx_depth': 10,
        'rx_range': 1000,
        'depth': 25
    }
    for k, v in env.items():
        assert k in e.keys(), 'Unknown key: '+k
        e[k] = _np.asarray(v, dtype=_np.float) if _np.size(v) > 1 else v
    max_range = _np.max(e['rx_range'])
    if _np.size(e['depth']) > 1:
        assert e['depth'].ndim == 2, 'depth must be a scalar or a Nx2 array'
        assert e['depth'].shape[1] == 2, 'depth must be a scalar or a Nx2 array'
        assert e['depth'][0,0] == 0, 'First range in depth array must be 0 m'
        assert e['depth'][-1,0] == max_range, 'Last range in depth array must be equal to range: '+str(max_range)+' m'
        max_depth = _np.max(e['depth'][:,1])
    else:
        max_depth = e['depth']
    if _np.size(e['soundspeed']) > 1:
        assert e['soundspeed'].ndim == 2, 'soundspeed must be a scalar or a Nx2 array'
        assert e['soundspeed'].shape[1] == 2, 'soundspeed must be a scalar or a Nx2 array'
        assert e['soundspeed'][0,0] == 0, 'First depth in soundspeed array must be 0 m'
        assert e['soundspeed'][-1,0] == max_depth, 'Last depth in soundspeed array must be equal to water depth: '+str(max_depth)+' m'
    assert e['tx_depth'] <= max_depth, 'tx_depth cannot exceed water depth: '+str(max_depth)+' m'
    assert e['rx_depth'] <= max_depth, 'rx_depth cannot exceed water depth: '+str(max_depth)+' m'
    e['max_range'] = max_range
    e['max_depth'] = max_depth
    return e

def _bellhop(env, type):
    assert _os.system('which bellhop.exe') == 0, 'bellhop.exe not found, please install acoustic toolbox from http://oalib.hlsresearch.com/Modes/AcousticsToolbox/'
    # generate environment file
    fh, fname = _mkstemp(suffix='.env')
    _os.write(fh, "'arlpy_uwa'\n".encode())                         # name
    _os.write(fh, (str(env['frequency'])+"\n").encode())            # frequency
    _os.write(fh, "1\n'CVWT'\n".encode())                           # nmedia, sspopt
    max_depth = env['max_depth']
    _os.write(fh, ("1 0.0 "+str(max_depth)+"\n").encode())          # depth
    svp = env['soundspeed']
    if _np.size(svp) == 1:
        _os.write(fh, ("0.0 "+str(svp)+" /\n"+str(max_depth)+" "+str(svp)+" /\n").encode())
    else:
        for j in svp.shape[0]:
            _os.write(fh, (str(svp[j,0])+" "+str(svp[j,1])+" /\n").encode())
    depth = env['depth']
    if _np.size(depth) == 1:
        _os.write(fh, ("'A' "+str(env['bottom_roughness'])+"\n").encode())
    else:
        # TODO: create bathy file
        _os.write(fh, ("'A*' "+str(env['bottom_roughness'])+"\n").encode())

# fprintf(f,'%0.1f %0.1f 0.0 %0.4f %0.1f /\n',env.depth,env.bottom(1),env.bottom(2),env.bottom(3));
# fprintf(f,'%i\n',length(env.txdepth));
# fprintf(f,'%0.1f ',env.txdepth);
# fprintf(f,'/\n');
# fprintf(f,'%i\n',length(env.rxdepth));
# fprintf(f,'%0.1f ',env.rxdepth);
# fprintf(f,'/\n');
# fprintf(f,'%i\n',length(env.rxrange));
# fprintf(f,'%0.4f ',env.rxrange/1000);
# fprintf(f,'/\n');
# fprintf(f,'''%s''\n',model.stype);
# fprintf(f,'%i\n%0.1f %0.1f /\n',model.nBeams,-model.alpha,model.alpha);
# fprintf(f,'0.0 %0.1f %0.4f\n',1.01*max(env.depth),1.01*max(env.rxrange)/1000);
# err = [];

    _os.close(fh)
    # run bellhop
    fname_base = fname[:-4]
    rv = _os.system('bellhop.exe '+fname_base)
    _os.unlink(fname)
    assert rv == 0, 'Error running bellhop.exe'
    # load results file

def _jasa2007(env, type):
    assert type == 'A', 'jasa2007 model only supports arrivals'
    assert False, 'jasa2007 unimplemented'
    pass

def pm_simulate(env, type='arrivals', model='auto'):
    """Use an acoustic propagation model to simulate an underwater environment.
    """
    if model == 'bellhop':
        model = _bellhop
    elif model == 'jasa2007':
        model = _jasa2007
    elif model == 'auto':
        if type == 'arrivals' and _np.size(env['soundspeed']) == 1 and _np.size(env['depth']) == 1:
            model = _jasa2007
        else:
            model = _bellhop
    if type == 'arrivals':
        return model(env, 'A')
    elif type == 'eigenrays':
        return model(env, 'E')
    elif type == 'coherent':
        return model(env, 'C')
    elif type == 'incoherent':
        return model(env, 'I')
    elif type == 'semicoherent':
        return model(env, 'S')
