##############################################################################
#
# Copyright (c) 2018, Mandar Chitre
#
# This file is part of arlpy which is released under Simplified BSD License.
# See file LICENSE or go to http://www.opensource.org/licenses/BSD-3-Clause
# for full license details.
#
##############################################################################

"""Underwater acoustics propagation modeling toolbox."""

import os as _os
import numpy as _np
from tempfile import mkstemp as _mkstemp

def create_env2d(**kv):
    """Create a new 2D underwater environment.

    A basic environment is created with default values. To see all the parameters
    available and their default values:

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d()
    >>> pm.print_env(env)

    The environment parameters may be changed by passing keyword arguments
    or modified later using a dictionary notation:

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d(depth=40, soundspeed=1540)
    >>> pm.print_env(env)
    >>> env['depth'] = 25
    >>> env['bottom_soundspeed'] = 1800
    >>> pm.print_env(env)

    The default environment has a constant sound speed. A depth dependent sound speed
    profile be provided as a Nx2 array of (depth, sound speed):

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d(depth=20, soundspeed=[[0,1540], [5,1535], [20,1530]])

    The default environment has a constant water depth. A range dependent bathymetry
    can be provided as a Nx2 array of (range, water depth):

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d(depth=[[0,20], [300,10], [500,18], [1000,15]])
    """
    env = {
        'name': 'arlpy',
        'frequency': 25000,           # Hz
        'soundspeed': 1500,           # m/s
        'bottom_soundspeed': 1600,    # m/s
        'bottom_density': 1.6,        # kg/m^3
        'bottom_absorption': 0.1,     # dB/wavelength
        'bottom_roughness': 0,        # m (rms)
        'tx_depth': 5,                # m
        'rx_depth': 10,               # m
        'rx_range': 1000,             # m
        'depth': 25                   # m
    }
    for k, v in kv.items():
        assert k in env.keys(), 'Unknown key: '+k
        env[k] = _np.asarray(v, dtype=_np.float) if _np.size(v) > 1 else v
    check_env2d(env)
    return env

def check_env2d(env):
    """Check the validity of a 2D underwater environment definition.

    :param env: environment definition

    Exceptions are thrown with appropriate error messages if the environment is invalid.

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d()
    >>> check_env2d(env)
    """
    max_range = _np.max(env['rx_range'])
    if _np.size(env['depth']) > 1:
        assert env['depth'].ndim == 2, 'depth must be a scalar or a Nx2 array'
        assert env['depth'].shape[1] == 2, 'depth must be a scalar or a Nx2 array'
        assert env['depth'][0,0] == 0, 'First range in depth array must be 0 m'
        assert env['depth'][-1,0] == max_range, 'Last range in depth array must be equal to range: '+str(max_range)+' m'
        max_depth = _np.max(env['depth'][:,1])
    else:
        max_depth = env['depth']
    if _np.size(env['soundspeed']) > 1:
        assert env['soundspeed'].ndim == 2, 'soundspeed must be a scalar or a Nx2 array'
        assert env['soundspeed'].shape[1] == 2, 'soundspeed must be a scalar or a Nx2 array'
        assert env['soundspeed'][0,0] == 0, 'First depth in soundspeed array must be 0 m'
        assert env['soundspeed'][-1,0] == max_depth, 'Last depth in soundspeed array must be equal to water depth: '+str(max_depth)+' m'
    assert env['tx_depth'] <= max_depth, 'tx_depth cannot exceed water depth: '+str(max_depth)+' m'
    assert env['rx_depth'] <= max_depth, 'rx_depth cannot exceed water depth: '+str(max_depth)+' m'

def print_env(env):
    """Display the environment in a human readable form.

    :param env: environment definition

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d(depth=40, soundspeed=1540)
    >>> pm.print_env(env)
    """
    keys = ['name'] + sorted(list(env.keys()-['name']))
    for k in keys:
        v = str(env[k])
        if '\n' in v:
            v = v.split('\n')
            print('%20s : '%(k) + v[0])
            for v1 in v[1:]:
                print('%20s   '%('') + v1)
        else:
            print('%20s : '%(k) + v)

def _print(fh, s):
    _os.write(fh, s.encode())

def _println(fh, s):
    _os.write(fh, (s+'\n').encode())

def _printarr(fh, a):
    if _np.size(a) == 1:
        _println(fh, "1")
        _println(fh, "%0.1f /" % (a))
    else:
        _println(fh, str(_np.size(a)))
        for j in a:
            _print(fh, "%0.1f " % (j))
        _println(fh, "/")

def _unlink(f):
    try:
        os.unlink(f)
    except:
        pass

def _bellhop(env, type, debug=False):
    # generate environment file
    fh, fname = _mkstemp(suffix='.env')
    fname_base = fname[:-4]
    _println(fh, "'"+env['name']+"'")
    _println(fh, "%0.1f" % (env['frequency']))
    _println(fh, "1")
    _println(fh, "'CVWT'")
    max_depth = env['max_depth']
    _println(fh, "1 0.0 %0.1f" % (max_depth))
    svp = env['soundspeed']
    if _np.size(svp) == 1:
        _println(fh, "0.0 %0.1f /" % (svp))
        _println(fh, "%0.1f %0.1f /" % (max_depth, svp))
    else:
        for j in svp.shape[0]:
            _println(fh, "%0.1f %0.1f /" % (svp[j,0], svp[j,1]))
    depth = env['depth']
    if _np.size(depth) == 1:
        _println(fh, "'A' %0.3f" % (env['bottom_roughness']))
    else:
        _println(fh, "'A*' %0.3f" % (env['bottom_roughness']))
        with open(fname_base+'.bty', 'wt') as f:
            f.write("'L'\n")
            f.write(str(_np.size(depth))+"\n")
            for j in depth.shape[0]:
                f.write("%0.4f %0.1f\n" % (depth[j,0]/1000, depth[j,1]))
    _println(fh, "%0.1f %0.1f 0.0 %0.4f %0.1f /" % (max_depth, env['bottom_soundspeed'], env['bottom_density'], env['bottom_absorption']))
    _printarr(fh, env['tx_depth'])
    _printarr(fh, env['rx_depth'])
    _printarr(fh, env['rx_range']/1000)
    _println(fh, "'"+type+"'")
    _println(fh, "0")
    _println(fh, "-89.0 89.0 /")
    _println(fh, "0.0 %0.1f %0.4f" % (1.01*max_depth, 1.01*env['max_range']/1000))
    _os.close(fh)
    # run bellhop
    rv = _os.system('bellhop.exe '+fname_base)
    if debug:
        print('[DEBUG] Bellhop files not deleted: '+fname_base+'.*')
    else:
        _unlink(fname)
        _unlink(fname_base+'.bty')
        _unlink(fname_base+'.prt')
    assert rv == 0, 'Error running bellhop.exe - please install acoustic toolbox from http://oalib.hlsresearch.com/Modes/AcousticsToolbox/'
    # load results file
    # TODO
    if not debug:
        _unlink(fname_base+'.arr')
        _unlink(fname_base+'.ray')
        _unlink(fname_base+'.shd')

def _jasa2007(env, type):
    assert type == 'A', 'jasa2007 model only supports arrivals'
    assert False, 'jasa2007 unimplemented'
    pass

def pm_simulate(env, type='arrivals', model='auto', debug=False):
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
        return model(env, 'A', debug)
    elif type == 'eigenrays':
        return model(env, 'E', debug)
    elif type == 'coherent':
        return model(env, 'C', debug)
    elif type == 'incoherent':
        return model(env, 'I', debug)
    elif type == 'semicoherent':
        return model(env, 'S', debug)
