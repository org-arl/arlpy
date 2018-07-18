##############################################################################
#
# Copyright (c) 2018, Mandar Chitre
#
# This file is part of arlpy which is released under Simplified BSD License.
# See file LICENSE or go to http://www.opensource.org/licenses/BSD-3-Clause
# for full license details.
#
##############################################################################

"""Underwater acoustics propagation modeling toolbox.

This toolbox currently uses the Bellhop acoustic propagation model. For this model
to work, the `acoustic toolbox <http://oalib.hlsresearch.com/Modes/AcousticsToolbox/>`_
must be installed on your computer and `bellhop.exe` should be in your PATH.
"""

import os as _os
import re as _re
import subprocess as _proc
import numpy as _np
import pandas as _pd
from tempfile import mkstemp as _mkstemp
import arlpy.plot as _plt
import bokeh as _bokeh

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
        'depth': 25,                  # m
        'min_angle': -0.9*_np.pi/2,   # rad
        'max_angle': 0.9*_np.pi/2     # rad
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
        assert env['min_angle'] > -_np.pi/2 and env['min_angle'] < _np.pi/2, 'min_angle must be in range (-pi/2, pi/2)'
        assert env['max_angle'] > -_np.pi/2 and env['max_angle'] < _np.pi/2, 'max_angle must be in range (-pi/2, pi/2)'
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

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d()
    >>> arrivals = pm.compute_arrivals(env)
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

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d()
    >>> rays = pm.compute_eigenrays(env)
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

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d()
    >>> rays = pm.compute_rays(env)
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

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d()
    >>> tloss = pm.compute_transmission_loss(env, mode='incoherent')
    """
    if mode not in ['coherent', 'incoherent', 'semicoherent']:
        raise ValueError('Unknown transmission loss mode: '+mode)
    if _np.size(env['tx_depth']) > 1:
        env = env.copy()
        env['tx_depth'] = env['tx_depth'][tx_depth_ndx]
    model = _select_model(env, mode, model)
    return model.run(env, mode, debug)

def arrivals_to_impulse_response(arrivals, fs, abs_time=False):
    """Convert arrival times and coefficients to an impulse response.

    :param arrivals: arrivals times (s) and coefficients
    :param fs: sampling rate (Hz)
    :param abs_time: absolute time (True) or relative time (False)
    :returns: impulse response

    If `abs_time` is set to True, the impulse response is placed such that
    the zero time corresponds to the time of transmission of signal.

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d()
    >>> arrivals = pm.compute_arrivals(env)
    >>> ir = pm.arrivals_to_impulse_response(arrivals, fs=192000)
    """
    t0 = 0 if abs_time else min(arrivals.time_of_arrival)
    irlen = int(_np.ceil((max(arrivals.time_of_arrival)-t0)*fs))+1
    ir = _np.zeros(irlen, dtype=_np.complex)
    for _, row in arrivals.iterrows():
        ndx = int(_np.round((row.time_of_arrival.real-t0)*fs))
        ir[ndx] = row.arrival_amplitude
    return ir

def plot_arrivals(arrivals, color='blue', **kwargs):
    """Plots the arrival times and amplitudes.

    :param arrivals: arrivals times (s) and coefficients
    :param color: line color (see `Bokeh colors <https://bokeh.pydata.org/en/latest/docs/reference/colors.html>`_)

    Other keyword arguments applicable for `arlpy.plot.figure()` are also supported.

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d()
    >>> arrivals = pm.compute_arrivals(env)
    >>> ir = pm.plot_arrivals(arrivals, color='red', width=800)
    """
    t0 = min(arrivals.time_of_arrival)
    t1 = max(arrivals.time_of_arrival)
    with _plt.figure(xlabel='Arrival time (s)', ylabel='Amplitude', **kwargs):
        _plt.plot([t0, t1], [0, 0], color=color)
        for _, row in arrivals.iterrows():
            t = row.time_of_arrival.real
            _plt.plot([t, t], [0, _np.abs(row.arrival_amplitude)], color=color)

def plot_rays(rays, **kwargs):
    """Plots ray paths.

    :param rays: ray paths

    Other keyword arguments applicable for `arlpy.plot.figure()` are also supported.

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d()
    >>> rays = pm.compute_eigenrays(env)
    >>> ir = pm.plot_rays(rays, width=1000)
    """
    rays = rays.sort_values('bottom_bounces', ascending=False)
    max_amp = _np.max(_np.abs(rays.bottom_bounces))
    with _plt.figure(xlabel='Range (m)', ylabel='Depth (m)', **kwargs):
        for _, row in rays.iterrows():
            c = int(255*_np.abs(row.bottom_bounces)/max_amp)
            c = _bokeh.colors.RGB(c, c, c)
            _plt.plot(row.ray[:,0], -row.ray[:,1], color=c)

def _select_model(env, task, model):
    if model is not None:
        if model == 'bellhop':
            return _Bellhop()
        raise ValueError('Unknown model: '+model)
    for m in [_Bellhop()]:
        if m.supports(env, task):
            return m
    raise ValueError('No suitable propagation model available')

### Bellhop propagation model ###

class _Bellhop:

    def __init__(self):
        pass

    def supports(self, env, task):
        if task == 'coherent' or task == 'incoherent' or task == 'semicoherent':
            return False
        return self._bellhop()

    def run(self, env, task, debug=False):
        taskmap = {
            'arrivals':     ['A', self._load_arrivals],
            'eigenrays':    ['E', self._load_rays],
            'rays':         ['R', self._load_rays],
            'coherent':     ['C', None],
            'incoherent':   ['I', None],
            'semicoherent': ['S', None]
        }
        fname_base = self._create_env_file(env, taskmap[task][0])
        if self._bellhop(fname_base):
            results = taskmap[task][1](fname_base)
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
            _proc.check_output(['bellhop.exe'] + list(args), stderr=_proc.STDOUT)
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

    def _create_env_file(self, env, taskcode):
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
            for j in range(svp.shape[0]):
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
        self._print(fh, "'"+taskcode+"'")
        self._print(fh, "0")
        self._print(fh, "%0.1f %0.1f /" % (env['min_angle']*180/_np.pi, env['max_angle']*180/_np.pi))
        self._print(fh, "0.0 %0.1f %0.4f" % (1.01*max_depth, 1.01*_np.max(env['rx_range'])/1000))
        _os.close(fh)
        return fname_base

    def _create_bty_file(self, filename, depth):
        with open(filename, 'wt') as f:
            f.write("'L'\n")
            f.write(str(_np.size(depth))+"\n")
            for j in range(depth.shape[0]):
                f.write("%0.4f %0.1f\n" % (depth[j,0]/1000, depth[j,1]))

    def _readf(self, f, types):
        p = _re.split(r' +', f.readline().strip())
        for j in range(len(p)):
            if len(types) > j:
                p[j] = types[j](p[j])
        return tuple(p)

    def _load_arrivals(self, fname_base):
        with open(fname_base+'.arr', 'rt') as f:
            freq, tx_depth_count, rx_depth_count, rx_range_count = self._readf(f, (float, int, int, int))
            tx_depth = self._readf(f, (float,)*tx_depth_count)
            rx_depth = self._readf(f, (float,)*rx_depth_count)
            rx_range = self._readf(f, (float,)*rx_range_count)
            arrivals = []
            for j in range(tx_depth_count):
                f.readline()
                for k in range(rx_depth_count):
                    for m in range(rx_range_count):
                        count = int(f.readline())
                        for n in range(count):
                            data = self._readf(f, (float, float, float, float, float, float, int, int))
                            arrivals.append(_pd.DataFrame({
                                'tx_depth_ndx': [j],
                                'rx_depth_ndx': [k],
                                'rx_range_ndx': [m],
                                'tx_depth': [tx_depth[j]],
                                'rx_depth': [rx_depth[k]],
                                'rx_range': [rx_range[m]],
                                'arrival_number': [n],
                                'arrival_amplitude': [data[0]*_np.exp(1j*data[1])],
                                'time_of_arrival': [data[2]],
                                'angle_of_departure': [data[4]*_np.pi/180],
                                'angle_of_arrival': [data[5]*_np.pi/180],
                                'surface_bounces': [data[6]],
                                'bottom_bounces': [data[7]]
                            }, index=[len(arrivals)+1]))
        return _pd.concat(arrivals)

    def _load_rays(self, fname_base):
        with open(fname_base+'.ray', 'rt') as f:
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            rays = []
            while True:
                s = f.readline()
                if s is None or len(s.strip()) == 0:
                    break
                a = float(s)
                pts, sb, bb = self._readf(f, (int, int, int))
                ray = _np.empty((pts, 2))
                for k in range(pts):
                    ray[k,:] = self._readf(f, (float, float))
                rays.append(_pd.DataFrame({
                    'angle_of_departure': [a*_np.pi/180],
                    'surface_bounces': [sb],
                    'bottom_bounces': [bb],
                    'ray': [ray]
                }))
        return _pd.concat(rays)
