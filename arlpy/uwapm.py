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
import subprocess as _proc
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
        if k not in env.keys():
            raise KeyError('Unknown key: '+k)
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
    try:
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
    except AssertionError as e:
        raise ValueError(e.args)

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

def compute_arrivals(env, model=None, debug=False):
    """Compute arrivals between each transmitter and receiver.

    :param env: environment definition
    :param model: propagation model to use (None to auto-select)
    :param debug: generate debug information for propagation model
    :returns: arrival times and coefficients for all transmitter-receiver combinations
    """
    model = _select_model(env, 'arrivals', model)
    return model.run(env, 'arrivals', debug)

def compute_eigenrays(env, tx_depth_ndx=0, rx_depth_ndx=0, rx_range_ndx=0, model=None, debug=False):
    """Compute eigenrays between a given transmitter and receiver.

    :param env: environment definition
    :param tx_depth_ndx: transmitter depth index
    :param rx_depth_ndx: receiver depth index
    :param rx_range_ndx: receiver range index
    :param model: propagation model to use (None to auto-select)
    :param debug: generate debug information for propagation model
    :returns: eigenrays paths
    """
    env = env.copy()
    if _np.size(env['tx_depth']) > 1:
        env['tx_depth'] = env['tx_depth'][tx_depth_ndx]
    if _np.size(env['rx_depth']) > 1:
        env['rx_depth'] = env['rx_depth'][rx_depth_ndx]
    if _np.size(env['rx_range']) > 1:
        env['rx_range'] = env['rx_range'][rx_range_ndx]
    model = _select_model(env, 'eigenrays', model)
    return model.run(env, 'eigenrays', debug)

def compute_rays(env, tx_depth_ndx=0, model=None, debug=False):
    """Compute rays from a given transmitter.

    :param env: environment definition
    :param tx_depth_ndx: transmitter depth index
    :param model: propagation model to use (None to auto-select)
    :param debug: generate debug information for propagation model
    :returns: ray paths
    """
    if _np.size(env['tx_depth']) > 1:
        env = env.copy()
        env['tx_depth'] = env['tx_depth'][tx_depth_ndx]
    model = _select_model(env, 'rays', model)
    return model.run(env, 'rays', debug)

def compute_transmission_loss(env, tx_depth_ndx=0, mode='coherent', model=None, debug=False):
    """Compute transmission loss from a given transmitter to all receviers.

    :param env: environment definition
    :param tx_depth_ndx: transmitter depth index
    :param mode: 'coherent', 'incoherent' or 'semicoherent'
    :param model: propagation model to use (None to auto-select)
    :param debug: generate debug information for propagation model
    :returns: transmission loss in dB at each receiver depth and range
    """
    if mode not in ['coherent', 'incoherent', 'semicoherent']:
        raise ValueError('Unknown transmission loss mode: '+mode)
    if _np.size(env['tx_depth']) > 1:
        env = env.copy()
        env['tx_depth'] = env['tx_depth'][tx_depth_ndx]
    model = _select_model(env, mode, model)
    return model.run(env, mode, debug)

def arrivals_to_impulse_response(arrivals, fs):
    """Convert arrival times and coefficients to an impulse response.

    :param arrivals: list of arrivals times (s) and coefficients
    :param fs: sampling rate (Hz)
    :returns: impulse response
    """
    pass

def _select_model(env, task, model):
    if model is not None:
        if model == 'bellhop':
            return _Bellhop()
        raise ValueError('Unknown model: '+model)
    for m in [_Bellhop()]:
        if m.supports(env, task):
            return m
    raise ValueError('No suitable propagation model available')

class _Bellhop:

    def __init__(self):
        self.executable = 'bellhop.exe'
        self.taskmap = {
            'arrivals': 'A',
            'eigenrays': 'E',
            'rays': 'R',
            'coherent': 'C',
            'incoherent': 'I',
            'semicoherent': 'S'
        }

    def supports(self, env, task):
        return self._bellhop()

    def run(self, env, task, debug=False):
        fname_base = self._create_env_file(env, task)
        if self._bellhop(fname_base):
            results = True # TODO
        else:
            results = None
        if debug:
            print('[DEBUG] Bellhop working files: '+fname_base+'.*')
        else:
            self._unlink(fname_base+'.env')
            self._unlink(fname_base+'.bty')
            self._unlink(fname_base+'.prt')
            self._unlink(fname_base+'.arr')
            self._unlink(fname_base+'.ray')
            self._unlink(fname_base+'.shd')
        return results

    def _bellhop(self, *args):
        try:
            _proc.check_output([self.executable] + list(args), stderr=_proc.STDOUT)
        except OSError:
            return False
        return True

    def _unlink(self, f):
        try:
            os.unlink(f)
        except:
            pass

    def _print(self, fh, s, newline=True):
        _os.write(fh, (s+'\n' if newline else s).encode())

    def _print_array(self, fh, a):
        if _np.size(a) == 1:
            self._print(fh, "1")
            self._print(fh, "%0.1f /" % (a))
        else:
            self._print(fh, str(_np.size(a)))
            for j in a:
                self._print(fh, "%0.1f " % (j), newline=False)
            self._print(fh, "/")

    def _create_env_file(self, env, task):
        task = self.taskmap[task]
        fh, fname = _mkstemp(suffix='.env')
        fname_base = fname[:-4]
        self._print(fh, "'"+env['name']+"'")
        self._print(fh, "%0.1f" % (env['frequency']))
        self._print(fh, "1")
        self._print(fh, "'CVWT'")
        max_depth = env['depth'] if _np.size(env['depth']) == 1 else _np.max(env['depth'][:,1])
        self._print(fh, "1 0.0 %0.1f" % (max_depth))
        svp = env['soundspeed']
        if _np.size(svp) == 1:
            self._print(fh, "0.0 %0.1f /" % (svp))
            self._print(fh, "%0.1f %0.1f /" % (max_depth, svp))
        else:
            for j in svp.shape[0]:
                self._print(fh, "%0.1f %0.1f /" % (svp[j,0], svp[j,1]))
        depth = env['depth']
        if _np.size(depth) == 1:
            self._print(fh, "'A' %0.3f" % (env['bottom_roughness']))
        else:
            self._print(fh, "'A*' %0.3f" % (env['bottom_roughness']))
            self._create_bty_file(fname_base+'.bty', depth)
        self._print(fh, "%0.1f %0.1f 0.0 %0.4f %0.1f /" % (max_depth, env['bottom_soundspeed'], env['bottom_density'], env['bottom_absorption']))
        self._print_array(fh, env['tx_depth'])
        self._print_array(fh, env['rx_depth'])
        self._print_array(fh, env['rx_range']/1000)
        self._print(fh, "'"+task+"'")
        self._print(fh, "0")
        self._print(fh, "-89.0 89.0 /")
        self._print(fh, "0.0 %0.1f %0.4f" % (1.01*max_depth, 1.01*_np.max(env['rx_range'])/1000))
        _os.close(fh)
        return fname_base

    def _create_bty_file(self, filename, depth):
        with open(filename, 'wt') as f:
            f.write("'L'\n")
            f.write(str(_np.size(depth))+"\n")
            for j in depth.shape[0]:
                f.write("%0.4f %0.1f\n" % (depth[j,0]/1000, depth[j,1]))
