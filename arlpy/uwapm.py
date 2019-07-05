##############################################################################
#
# Copyright (c) 2018, Mandar Chitre
#
# This file is part of arlpy which is released under Simplified BSD License.
# See file LICENSE or go to http://www.opensource.org/licenses/BSD-3-Clause
# for full license details.
#
##############################################################################

"""Underwater acoustic propagation modeling toolbox.

This toolbox currently uses the Bellhop acoustic propagation model. For this model
to work, the `acoustic toolbox <http://oalib.hlsresearch.com/Modes/AcousticsToolbox/>`_
must be installed on your computer and `bellhop.exe` should be in your PATH.

.. sidebar:: Sample Jupyter notebook

    For usage examples of this toolbox, see `Bellhop notebook <_static/bellhop.html>`_.
"""

import os as _os
import re as _re
import subprocess as _proc
import numpy as _np
from scipy import interpolate as _interp
import pandas as _pd
from tempfile import mkstemp as _mkstemp
from struct import unpack as _unpack
from sys import float_info as _fi
import arlpy.plot as _plt
import bokeh as _bokeh

# constants
linear = 'linear'
spline = 'spline'
curvilinear = 'curvilinear'
arrivals = 'arrivals'
eigenrays = 'eigenrays'
rays = 'rays'
coherent = 'coherent'
incoherent = 'incoherent'
semicoherent = 'semicoherent'

# models (in order of preference)
_models = []

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
        'type': '2D',                   # 2D/3D
        'frequency': 25000,             # Hz
        'soundspeed': 1500,             # m/s
        'soundspeed_interp': spline,    # spline/linear
        'bottom_soundspeed': 1600,      # m/s
        'bottom_density': 1600,         # kg/m^3
        'bottom_absorption': 0.1,       # dB/wavelength
        'bottom_roughness': 0,          # m (rms)
        'surface': None,                # surface profile
        'surface_interp': linear,       # curvilinear/linear
        'tx_depth': 5,                  # m
        'tx_directionality': None,      # [(deg, dB)...]
        'rx_depth': 10,                 # m
        'rx_range': 1000,               # m
        'depth': 25,                    # m
        'depth_interp': linear,         # curvilinear/linear
        'min_angle': -80,               # deg
        'max_angle': 80,                # deg
        'nbeams': 0                     # number of beams (0 = auto)
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
        assert env['type'] == '2D', 'Not a 2D environment'
        max_range = _np.max(env['rx_range'])
        if env['surface'] is not None:
            assert _np.size(env['surface']) > 1, 'surface must be an Nx2 array'
            assert env['surface'].ndim == 2, 'surface must be a scalar or an Nx2 array'
            assert env['surface'].shape[1] == 2, 'surface must be a scalar or an Nx2 array'
            assert env['surface'][0,0] <= 0, 'First range in surface array must be 0 m'
            assert env['surface'][-1,0] >= max_range, 'Last range in surface array must be beyond maximum range: '+str(max_range)+' m'
            assert _np.all(_np.diff(env['surface'][:,0]) > 0), 'surface array must be strictly monotonic in range'
            assert env['surface_interp'] == curvilinear or env['surface_interp'] == linear, 'Invalid interpolation type: '+str(env['surface_interp'])
        if _np.size(env['depth']) > 1:
            assert env['depth'].ndim == 2, 'depth must be a scalar or an Nx2 array'
            assert env['depth'].shape[1] == 2, 'depth must be a scalar or an Nx2 array'
            assert env['depth'][0,0] <= 0, 'First range in depth array must be 0 m'
            assert env['depth'][-1,0] >= max_range, 'Last range in depth array must be beyond maximum range: '+str(max_range)+' m'
            assert _np.all(_np.diff(env['depth'][:,0]) > 0), 'Depth array must be strictly monotonic in range'
            assert env['depth_interp'] == curvilinear or env['depth_interp'] == linear, 'Invalid interpolation type: '+str(env['depth_interp'])
            max_depth = _np.max(env['depth'][:,1])
        else:
            max_depth = env['depth']
        if _np.size(env['soundspeed']) > 1:
            assert env['soundspeed'].ndim == 2, 'soundspeed must be a scalar or an Nx2 array'
            assert env['soundspeed'].shape[1] == 2, 'soundspeed must be a scalar or an Nx2 array'
            assert env['soundspeed'].shape[0] > 3, 'soundspeed profile must have at least 4 points'
            assert env['soundspeed'][0,0] <= 0, 'First depth in soundspeed array must be 0 m'
            assert env['soundspeed'][-1,0] >= max_depth, 'Last depth in soundspeed array must be beyond water depth: '+str(max_depth)+' m'
            assert _np.all(_np.diff(env['soundspeed'][:,0]) > 0), 'Soundspeed array must be strictly monotonic in depth'
            assert env['soundspeed_interp'] == spline or env['soundspeed_interp'] == linear, 'Invalid interpolation type: '+str(env['soundspeed_interp'])
        assert _np.max(env['tx_depth']) <= max_depth, 'tx_depth cannot exceed water depth: '+str(max_depth)+' m'
        assert _np.max(env['rx_depth']) <= max_depth, 'rx_depth cannot exceed water depth: '+str(max_depth)+' m'
        assert env['min_angle'] > -90 and env['min_angle'] < 90, 'min_angle must be in range (-90, 90)'
        assert env['max_angle'] > -90 and env['max_angle'] < 90, 'max_angle must be in range (-90, 90)'
        if env['tx_directionality'] is not None:
            assert _np.size(env['tx_directionality']) > 1, 'tx_directionality must be an Nx2 array'
            assert env['tx_directionality'].ndim == 2, 'tx_directionality must be an Nx2 array'
            assert env['tx_directionality'].shape[1] == 2, 'tx_directionality must be an Nx2 array'
            assert _np.all(env['tx_directionality'][:,0] >= -180) and _np.all(env['tx_directionality'][:,0] <= 180), 'tx_directionality angles must be in [-90, 90]'
    except AssertionError as e:
        raise ValueError(e.args)

def print_env(env):
    """Display the environment in a human readable form.

    :param env: environment definition

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d(depth=40, soundspeed=1540)
    >>> pm.print_env(env)
    """
    check_env2d(env)
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

def plot_env(env, surface_color='dodgerblue', bottom_color='peru', tx_color='orangered', rx_color='midnightblue', rx_plot=None, **kwargs):
    """Plots a visual representation of the environment.

    :param env: environment description
    :param surface_color: color of the surface (see `Bokeh colors <https://bokeh.pydata.org/en/latest/docs/reference/colors.html>`_)
    :param bottom_color: color of the bottom (see `Bokeh colors <https://bokeh.pydata.org/en/latest/docs/reference/colors.html>`_)
    :param tx_color: color of transmitters (see `Bokeh colors <https://bokeh.pydata.org/en/latest/docs/reference/colors.html>`_)
    :param rx_color: color of receviers (see `Bokeh colors <https://bokeh.pydata.org/en/latest/docs/reference/colors.html>`_)
    :param rx_plot: True to plot all receivers, False to not plot any receivers, None to automatically decide

    Other keyword arguments applicable for `arlpy.plot.plot()` are also supported.

    The surface, bottom, transmitters (marker: '*') and receivers (marker: 'o')
    are plotted in the environment. If `rx_plot` is set to None and there are
    more than 2000 receivers, they are not plotted.

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d(depth=[[0, 40], [100, 30], [500, 35], [700, 20], [1000,45]])
    >>> pm.plot_env(env)
    """
    check_env2d(env)
    min_x = 0
    max_x = _np.max(env['rx_range'])
    if max_x-min_x > 10000:
        divisor = 1000
        min_x /= divisor
        max_x /= divisor
        xlabel = 'Range (km)'
    else:
        divisor = 1
        xlabel = 'Range (m)'
    if env['surface'] is None:
        min_y = 0
    else:
        min_y = _np.min(env['surface'][:,1])
    if _np.size(env['depth']) > 1:
        max_y = _np.max(env['depth'][:,1])
    else:
        max_y = env['depth']
    mgn_x = 0.01*(max_x-min_x)
    mgn_y = 0.1*(max_y-min_y)
    oh = _plt.hold()
    if env['surface'] is None:
        _plt.plot([min_x, max_x], [0, 0], xlabel=xlabel, ylabel='Depth (m)', xlim=(min_x-mgn_x, max_x+mgn_x), ylim=(-max_y-mgn_y, -min_y+mgn_y), color=surface_color, **kwargs)
    else:
        # linear and curvilinear options use the same altimetry, just with different normals
        s = env['surface']
        _plt.plot(s[:,0]/divisor, -s[:,1], xlabel=xlabel, ylabel='Depth (m)', xlim=(min_x-mgn_x, max_x+mgn_x), ylim=(-max_y-mgn_y, -min_y+mgn_y), color=surface_color, **kwargs)
    if _np.size(env['depth']) == 1:
        _plt.plot([min_x, max_x], [-env['depth'], -env['depth']], color=bottom_color)
    else:
        # linear and curvilinear options use the same bathymetry, just with different normals
        s = env['depth']
        _plt.plot(s[:,0]/divisor, -s[:,1], color=bottom_color)
    txd = env['tx_depth']
    _plt.plot([0]*_np.size(txd), -txd, marker='*', style=None, color=tx_color)
    if rx_plot is None:
        rx_plot = _np.size(env['rx_depth'])*_np.size(env['rx_range']) < 2000
    if rx_plot:
        rxr = env['rx_range']
        if _np.size(rxr) == 1:
            rxr = [rxr]
        for r in _np.array(rxr):
            rxd = env['rx_depth']
            _plt.plot([r/divisor]*_np.size(rxd), -rxd, marker='o', style=None, color=rx_color)
    _plt.hold(oh)

def plot_ssp(env, **kwargs):
    """Plots the sound speed profile.

    :param env: environment description

    Other keyword arguments applicable for `arlpy.plot.plot()` are also supported.

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d(soundspeed=[[ 0, 1540], [10, 1530], [20, 1532], [25, 1533], [30, 1535]])
    >>> pm.plot_ssp(env)
    """
    check_env2d(env)
    if _np.size(env['soundspeed']) == 1:
        if _np.size(env['depth']) > 1:
            max_y = _np.max(env['depth'][:,1])
        else:
            max_y = env['depth']
        _plt.plot([env['soundspeed'], env['soundspeed']], [0, -max_y], xlabel='Soundspeed (m/s)', ylabel='Depth (m)', **kwargs)
    elif env['soundspeed_interp'] == spline:
        s = env['soundspeed']
        ynew = _np.linspace(_np.min(s[:,0]), _np.max(s[:,0]), 100)
        tck = _interp.splrep(s[:,0], s[:,1], s=0)
        xnew = _interp.splev(ynew, tck, der=0)
        _plt.plot(xnew, -ynew, xlabel='Soundspeed (m/s)', ylabel='Depth (m)', hold=True, **kwargs)
        _plt.plot(s[:,1], -s[:,0], marker='.', style=None, **kwargs)
    else:
        s = env['soundspeed']
        _plt.plot(s[:,1], -s[:,0], xlabel='Soundspeed (m/s)', ylabel='Depth (m)', **kwargs)

def compute_arrivals(env, model=None, debug=False):
    """Compute arrivals between each transmitter and receiver.

    :param env: environment definition
    :param model: propagation model to use (None to auto-select)
    :param debug: generate debug information for propagation model
    :returns: arrival times and coefficients for all transmitter-receiver combinations

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d()
    >>> arrivals = pm.compute_arrivals(env)
    >>> pm.plot_arrivals(arrivals)
    """
    check_env2d(env)
    (model_name, model) = _select_model(env, arrivals, model)
    if debug:
        print('[DEBUG] Model: '+model_name)
    return model.run(env, arrivals, debug)

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
    >>> pm.plot_rays(rays, width=1000)
    """
    check_env2d(env)
    env = env.copy()
    if _np.size(env['tx_depth']) > 1:
        env['tx_depth'] = env['tx_depth'][tx_depth_ndx]
    if _np.size(env['rx_depth']) > 1:
        env['rx_depth'] = env['rx_depth'][rx_depth_ndx]
    if _np.size(env['rx_range']) > 1:
        env['rx_range'] = env['rx_range'][rx_range_ndx]
    (model_name, model) = _select_model(env, eigenrays, model)
    if debug:
        print('[DEBUG] Model: '+model_name)
    return model.run(env, eigenrays, debug)

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
    >>> pm.plot_rays(rays, width=1000)
    """
    check_env2d(env)
    if _np.size(env['tx_depth']) > 1:
        env = env.copy()
        env['tx_depth'] = env['tx_depth'][tx_depth_ndx]
    (model_name, model) = _select_model(env, rays, model)
    if debug:
        print('[DEBUG] Model: '+model_name)
    return model.run(env, rays, debug)

def compute_transmission_loss(env, tx_depth_ndx=0, mode=coherent, model=None, debug=False):
    """Compute transmission loss from a given transmitter to all receviers.

    :param env: environment definition
    :param tx_depth_ndx: transmitter depth index
    :param mode: coherent, incoherent or semicoherent
    :param model: propagation model to use (None to auto-select)
    :param debug: generate debug information for propagation model
    :returns: complex transmission loss at each receiver depth and range

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d()
    >>> tloss = pm.compute_transmission_loss(env, mode=pm.incoherent)
    >>> pm.plot_transmission_loss(tloss, width=1000)
    """
    check_env2d(env)
    if mode not in [coherent, incoherent, semicoherent]:
        raise ValueError('Unknown transmission loss mode: '+mode)
    if _np.size(env['tx_depth']) > 1:
        env = env.copy()
        env['tx_depth'] = env['tx_depth'][tx_depth_ndx]
    (model_name, model) = _select_model(env, mode, model)
    if debug:
        print('[DEBUG] Model: '+model_name)
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

def plot_arrivals(arrivals, dB=False, color='blue', **kwargs):
    """Plots the arrival times and amplitudes.

    :param arrivals: arrivals times (s) and coefficients
    :param dB: True to plot in dB, False for linear scale
    :param color: line color (see `Bokeh colors <https://bokeh.pydata.org/en/latest/docs/reference/colors.html>`_)

    Other keyword arguments applicable for `arlpy.plot.plot()` are also supported.

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d()
    >>> arrivals = pm.compute_arrivals(env)
    >>> pm.plot_arrivals(arrivals)
    """
    t0 = min(arrivals.time_of_arrival)
    t1 = max(arrivals.time_of_arrival)
    oh = _plt.hold()
    if dB:
        min_y = 20*_np.log10(_np.max(_np.abs(arrivals.arrival_amplitude)))-60
        ylabel = 'Amplitude (dB)'
    else:
        ylabel = 'Amplitude'
        _plt.plot([t0, t1], [0, 0], xlabel='Arrival time (s)', ylabel=ylabel, color=color, **kwargs)
        min_y = 0
    for _, row in arrivals.iterrows():
        t = row.time_of_arrival.real
        y = _np.abs(row.arrival_amplitude)
        if dB:
            y = max(20*_np.log10(_fi.epsilon+y), min_y)
        _plt.plot([t, t], [min_y, y], xlabel='Arrival time (s)', ylabel=ylabel, ylim=[min_y, min_y+70], color=color, **kwargs)
    _plt.hold(oh)

def plot_rays(rays, env=None, invert_colors=False, **kwargs):
    """Plots ray paths.

    :param rays: ray paths
    :param env: environment definition
    :param invert_colors: False to use black for high intensity rays, True to use white

    If environment definition is provided, it is overlayed over this plot using default
    parameters for `arlpy.uwapm.plot_env()`.

    Other keyword arguments applicable for `arlpy.plot.plot()` are also supported.

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d()
    >>> rays = pm.compute_eigenrays(env)
    >>> pm.plot_rays(rays, width=1000)
    """
    rays = rays.sort_values('bottom_bounces', ascending=False)
    max_amp = _np.max(_np.abs(rays.bottom_bounces)) if len(rays.bottom_bounces) > 0 else 0
    if max_amp <= 0:
        max_amp = 1
    divisor = 1
    xlabel = 'Range (m)'
    r = []
    for _, row in rays.iterrows():
        r += list(row.ray[:,0])
    if max(r)-min(r) > 10000:
        divisor = 1000
        xlabel = 'Range (km)'
    oh = _plt.hold()
    for _, row in rays.iterrows():
        c = int(255*_np.abs(row.bottom_bounces)/max_amp)
        if invert_colors:
            c = 255-c
        c = _bokeh.colors.RGB(c, c, c)
        _plt.plot(row.ray[:,0]/divisor, -row.ray[:,1], color=c, xlabel=xlabel, ylabel='Depth (m)', **kwargs)
    if env is not None:
        plot_env(env)
    _plt.hold(oh)

def plot_transmission_loss(tloss, env=None, **kwargs):
    """Plots transmission loss.

    :param tloss: complex transmission loss
    :param env: environment definition

    If environment definition is provided, it is overlayed over this plot using default
    parameters for `arlpy.uwapm.plot_env()`.

    Other keyword arguments applicable for `arlpy.plot.image()` are also supported.

    >>> import arlpy.uwapm as pm
    >>> import numpy as np
    >>> env = pm.create_env2d(
            rx_depth=np.arange(0, 25),
            rx_range=np.arange(0, 1000),
            min_angle=-45,
            max_angle=45
        )
    >>> tloss = pm.compute_transmission_loss(env)
    >>> pm.plot_transmission_loss(tloss, width=1000)
    """
    xr = (min(tloss.columns), max(tloss.columns))
    yr = (-max(tloss.index), -min(tloss.index))
    xlabel = 'Range (m)'
    if xr[1]-xr[0] > 10000:
        xr = (min(tloss.columns)/1000, max(tloss.columns)/1000)
        xlabel = 'Range (km)'
    oh = _plt.hold()
    _plt.image(20*_np.log10(_fi.epsilon+_np.abs(_np.flipud(_np.array(tloss)))), x=xr, y=yr, xlabel=xlabel, ylabel='Depth (m)', xlim=xr, ylim=yr, **kwargs)
    if env is not None:
        plot_env(env, rx_plot=False)
    _plt.hold(oh)

def models(env=None, task=None):
    """List available models.

    :param env: environment to model
    :param task: arrivals/eigenrays/rays/coherent/incoherent/semicoherent
    :returns: list of models that can be used

    >>> import arlpy.uwapm as pm
    >>> pm.models()
    ['bellhop']
    >>> env = pm.create_env2d()
    >>> pm.models(env, task=coherent)
    ['bellhop']
    """
    if env is not None:
        check_env2d(env)
    if (env is None and task is not None) or (env is not None and task is None):
        raise ValueError('env and task should be both specified together')
    rv = []
    for m in _models:
        if m[1]().supports(env, task):
            rv.append(m[0])
    return rv

def _select_model(env, task, model):
    if model is not None:
        for m in _models:
            if m[0] == model:
                return (m[0], m[1]())
        raise ValueError('Unknown model: '+model)
    for m in _models:
        mm = m[1]()
        if mm.supports(env, task):
            return (m[0], mm)
    raise ValueError('No suitable propagation model available')

### Bellhop propagation model ###

class _Bellhop:

    def __init__(self):
        pass

    def supports(self, env=None, task=None):
        if env is not None and env['type'] != '2D':
            return False
        fh, fname = _mkstemp(suffix='.env')
        _os.close(fh)
        fname_base = fname[:-4]
        self._unlink(fname_base+'.env')
        rv = self._bellhop(fname_base)
        self._unlink(fname_base+'.prt')
        self._unlink(fname_base+'.log')
        return rv

    def run(self, env, task, debug=False):
        taskmap = {
            arrivals:     ['A', self._load_arrivals],
            eigenrays:    ['E', self._load_rays],
            rays:         ['R', self._load_rays],
            coherent:     ['C', self._load_shd],
            incoherent:   ['I', self._load_shd],
            semicoherent: ['S', self._load_shd]
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
            self._unlink(fname_base+'.ati')
            self._unlink(fname_base+'.sbp')
            self._unlink(fname_base+'.prt')
            self._unlink(fname_base+'.log')
            self._unlink(fname_base+'.arr')
            self._unlink(fname_base+'.ray')
            self._unlink(fname_base+'.shd')
        return results

    def _bellhop(self, *args):
        try:
            _proc.call(['bellhop.exe'] + list(args), stderr=_proc.STDOUT)
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
            self._print(fh, "%0.4f /" % (a))
        else:
            self._print(fh, str(_np.size(a)))
            for j in a:
                self._print(fh, "%0.4f " % (j), newline=False)
            self._print(fh, "/")

    def _create_env_file(self, env, taskcode):
        fh, fname = _mkstemp(suffix='.env')
        fname_base = fname[:-4]
        self._print(fh, "'"+env['name']+"'")
        self._print(fh, "%0.4f" % (env['frequency']))
        self._print(fh, "1")
        if env['surface'] is None:
            self._print(fh, "'%cVWT'" % ('S' if env['soundspeed_interp'] == spline else 'C'))
        else:
            self._print(fh, "'%cVWT*'" % ('S' if env['soundspeed_interp'] == spline else 'C'))
            self._create_bty_ati_file(fname_base+'.ati', env['surface'], env['surface_interp'])
        max_depth = env['depth'] if _np.size(env['depth']) == 1 else _np.max(env['depth'][:,1])
        self._print(fh, "1 0.0 %0.4f" % (max_depth))
        svp = env['soundspeed']
        if _np.size(svp) == 1:
            self._print(fh, "0.0 %0.4f /" % (svp))
            self._print(fh, "%0.4f %0.4f /" % (max_depth, svp))
        else:
            for j in range(svp.shape[0]):
                self._print(fh, "%0.4f %0.4f /" % (svp[j,0], svp[j,1]))
        depth = env['depth']
        if _np.size(depth) == 1:
            self._print(fh, "'A' %0.4f" % (env['bottom_roughness']))
        else:
            self._print(fh, "'A*' %0.4f" % (env['bottom_roughness']))
            self._create_bty_ati_file(fname_base+'.bty', depth, env['depth_interp'])
        self._print(fh, "%0.4f %0.4f 0.0 %0.4f %0.4f /" % (max_depth, env['bottom_soundspeed'], env['bottom_density']/1000, env['bottom_absorption']))
        self._print_array(fh, env['tx_depth'])
        self._print_array(fh, env['rx_depth'])
        self._print_array(fh, env['rx_range']/1000)
        if env['tx_directionality'] is None:
            self._print(fh, "'"+taskcode+"'")
        else:
            self._print(fh, "'"+taskcode+" *'")
            self._create_sbp_file(fname_base+'.sbp', env['tx_directionality'])
        self._print(fh, "%d" % (env['nbeams']))
        self._print(fh, "%0.4f %0.4f /" % (env['min_angle'], env['max_angle']))
        self._print(fh, "0.0 %0.4f %0.4f" % (1.01*max_depth, 1.01*_np.max(env['rx_range'])/1000))
        _os.close(fh)
        return fname_base

    def _create_bty_ati_file(self, filename, depth, interp):
        with open(filename, 'wt') as f:
            f.write("'%c'\n" % ('C' if interp == curvilinear else 'L'))
            f.write(str(depth.shape[0])+"\n")
            for j in range(depth.shape[0]):
                f.write("%0.4f %0.4f\n" % (depth[j,0]/1000, depth[j,1]))

    def _create_sbp_file(self, filename, dir):
        with open(filename, 'wt') as f:
            f.write(str(dir.shape[0])+"\n")
            for j in range(dir.shape[0]):
                f.write("%0.4f %0.4f\n" % (dir[j,0], dir[j,1]))

    def _readf(self, f, types, dtype=str):
        if type(f) is str:
            p = _re.split(r' +', f.strip())
        else:
            p = _re.split(r' +', f.readline().strip())
        for j in range(len(p)):
            if len(types) > j:
                p[j] = types[j](p[j])
            else:
                p[j] = dtype(p[j])
        return tuple(p)

    def _load_arrivals(self, fname_base):
        with open(fname_base+'.arr', 'rt') as f:
            hdr = f.readline()
            if hdr.find('2D') >= 0:
                freq = self._readf(f, (float,))
                tx_depth_info = self._readf(f, (int,), float)
                tx_depth_count = tx_depth_info[0]
                tx_depth = tx_depth_info[1:]
                assert tx_depth_count == len(tx_depth)
                rx_depth_info = self._readf(f, (int,), float)
                rx_depth_count = rx_depth_info[0]
                rx_depth = rx_depth_info[1:]
                assert rx_depth_count == len(rx_depth)
                rx_range_info = self._readf(f, (int,), float)
                rx_range_count = rx_range_info[0]
                rx_range = rx_range_info[1:]
                assert rx_range_count == len(rx_range)
            else:
                freq, tx_depth_count, rx_depth_count, rx_range_count = self._readf(hdr, (float, int, int, int))
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
                                'angle_of_departure': [data[4]],
                                'angle_of_arrival': [data[5]],
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
                    'angle_of_departure': [a],
                    'surface_bounces': [sb],
                    'bottom_bounces': [bb],
                    'ray': [ray]
                }))
        return _pd.concat(rays)

    def _load_shd(self, fname_base):
        with open(fname_base+'.shd', 'rb') as f:
            recl, = _unpack('i', f.read(4))
            title = str(f.read(80))
            f.seek(4*recl, 0)
            ptype = f.read(10).decode('utf8').strip()
            assert ptype == 'rectilin', 'Invalid file format (expecting ptype == "rectilin")'
            f.seek(8*recl, 0)
            nfreq, ntheta, nsx, nsy, nsd, nrd, nrr, atten = _unpack('iiiiiiif', f.read(32))
            assert nfreq == 1, 'Invalid file format (expecting nfreq == 1)'
            assert ntheta == 1, 'Invalid file format (expecting ntheta == 1)'
            assert nsd == 1, 'Invalid file format (expecting nsd == 1)'
            f.seek(32*recl, 0)
            pos_r_depth = _unpack('f'*nrd, f.read(4*nrd))
            f.seek(36*recl, 0)
            pos_r_range = _unpack('f'*nrr, f.read(4*nrr))
            pressure = _np.zeros((nrd, nrr), dtype=_np.complex)
            for ird in range(nrd):
                recnum = 10 + ird
                f.seek(recnum*4*recl, 0)
                temp = _np.array(_unpack('f'*2*nrr, f.read(2*nrr*4)))
                pressure[ird,:] = temp[::2] + 1j*temp[1::2]
        return _pd.DataFrame(pressure, index=pos_r_depth, columns=pos_r_range)

_models.append(('bellhop', _Bellhop))
