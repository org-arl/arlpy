##############################################################################
#
# Copyright (c) 2018, Mandar Chitre
#
# This file is part of arlpy which is released under Simplified BSD License.
# See file LICENSE or go to http://www.opensource.org/licenses/BSD-3-Clause
# for full license details.
#
##############################################################################

"""Plotting utility functions for Jupyter notebooks using Bokeh."""

import numpy as _np
import bokeh.plotting as _bplt
import bokeh.models as _bmodels
import bokeh.palettes as _bpal
import scipy.signal as _sig
from warnings import warn as _warn

_bplt.output_notebook(hide_banner=True)

_figure = None
_hold = False
_figsize = [600, 400]

def _new_figure(title, width, height, xlabel, ylabel, xlim, ylim):
    if width is None:
        width = _figsize[0]
    if height is None:
        height = _figsize[1]
    return _bplt.figure(title=title, plot_width=width, plot_height=height, x_range=xlim, y_range=ylim, x_axis_label=xlabel, y_axis_label=ylabel)

def figsize(x, y):
    """Sets the default figure size in pixels."""
    global _figsize
    _figsize = [x, y]

def hold(enable):
    """Combines multiple plots into one.

    >>> from arlpy.plot import hold, plot
    >>> hold(True)
    >>> plot([0,10], [0,10], color='blue', legend='A')
    >>> plot([10,0], [0,10], marker='o', color='green', legend='B')
    >>> hold(False)
    """
    global _hold, _figure
    _hold = enable
    if not _hold and _figure is not None:
        _bplt.show(_figure)
        _figure = None

def figure(title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, width=None, height=None):
    """Creates a new figure.

    >>> from arlpy.plot import figure, plot
    >>> figure(title='Demo', width=500)
    >>> plot([0,10], [0,10])
    """
    global _figure
    _figure = _new_figure(title, width, height, xlabel, ylabel, xlim, ylim)
    return _figure

def plot(x, y=None, fs=None, maxpts=10000, color='blue', marker=None, filled=False, size=6, mskip=0, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, width=None, height=None, legend=None, hold=False):
    """Plots a line graph or time series.

    :param x: x data or time series data (if y is None)
    :param y: y data or None (if time series)
    :param fs: sampling rate for time series data
    :param maxpts: maximum number of points to plot (downsampled if more points provided)
    :param color: line color (see [bokeh colors](https://bokeh.pydata.org/en/latest/docs/reference/colors.html))
    :param marker: point markers (see scatter())
    :param filled: filled markers or outlined ones
    :param size: marker size
    :param mskip: number of points to skip marking (to avoid too many markers)
    :param title: figure title
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param xlim: x-axis limits (min, max)
    :param ylim: y-axis limits (min, max)
    :param width: figure width in pixels
    :param height: figure height in pixels
    :param legend: legend text
    :param hold: if set to True, output is not plotted immediately, but combined with the next plot

    >>> from arlpy.plot import plot
    >>> import numpy as np
    >>> plot([0,10], [1,-1], color='blue', marker='o', filled=True, legend='A', hold=True)
    >>> plot(np.random.normal(size=1000), fs=100, color='green', legend='B')
    """
    global _figure
    x = _np.asarray(x, dtype=_np.float)
    if y is None:
        y = x
        x = _np.arange(x.size)
        if fs is not None:
            x = x/fs
            if xlabel is None:
                xlabel = 'Time (s)'
        if xlim is None:
            xlim = (x[0], x[-1])
    else:
        y = _np.asarray(y, dtype=_np.float)
    if _figure is None:
        _figure = _new_figure(title, width, height, xlabel, ylabel, xlim, ylim)
    if x.size > maxpts:
        n = int(_np.ceil(x.size/maxpts))
        x = x[::n]
        y = y[::n]
        _figure.add_layout(_bmodels.Label(x=5, y=5, x_units='screen', y_units='screen', text='Downsampled by '+str(n), text_font_size="8pt", text_alpha=0.5))
    _figure.line(x, y, line_color=color, legend=legend)
    if marker is not None:
        scatter(x[::(mskip+1)], y[::(mskip+1)], marker=marker, filled=filled, size=size, color=color, legend=legend, hold=True)
    if not hold and not _hold:
        _bplt.show(_figure)
        _figure = None

def scatter(x, y, marker='o', filled=False, size=6, color='blue', title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, width=None, height=None, legend=None, hold=False):
    """Plots a scatter plot.

    :param x: x data
    :param y: y data
    :param color: line color (see [Bokeh colors](https://bokeh.pydata.org/en/latest/docs/reference/colors.html))
    :param marker: point markers (see scatter())
    :param filled: filled markers or outlined ones
    :param size: marker size
    :param title: figure title
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param xlim: x-axis limits (min, max)
    :param ylim: y-axis limits (min, max)
    :param width: figure width in pixels
    :param height: figure height in pixels
    :param legend: legend text
    :param hold: if set to True, output is not plotted immediately, but combined with the next plot

    >>> from arlpy.plot import scatter
    >>> import numpy as np
    >>> scatter(np.random.normal(size=100), np.random.normal(size=100), color='blue', marker='o')
    """
    global _figure
    if _figure is None:
        _figure = _new_figure(title, width, height, xlabel, ylabel, xlim, ylim)
    x = _np.asarray(x, dtype=_np.float)
    y = _np.asarray(y, dtype=_np.float)
    if marker == 'o':
        _figure.circle(x, y, size=size, line_color=color, fill_color=color if filled else None, legend=legend)
    elif marker == 's':
        _figure.square(x, y, size=size, line_color=color, fill_color=color if filled else None, legend=legend)
    elif marker == '*':
        _figure.asterisk(x, y, size=size, line_color=color, fill_color=color if filled else None, legend=legend)
    elif marker == 'x':
        _figure.x(x, y, size=size, line_color=color, fill_color=color if filled else None, legend=legend)
    elif marker == '+':
        _figure.cross(x, y, size=size, line_color=color, fill_color=color if filled else None, legend=legend)
    elif marker == 'd':
        _figure.diamond(x, y, size=size, line_color=color, fill_color=color if filled else None, legend=legend)
    elif marker == '^':
        _figure.triangle(x, y, size=size, line_color=color, fill_color=color if filled else None, legend=legend)
    elif marker is not None:
        _warn('Bad marker type: '+marker)
    if not hold and not _hold:
        _bplt.show(_figure)
        _figure = None

def image(img, x=None, y=None, colormap='Plasma256', clim=None, clabel=None, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, width=None, height=None, hold=False):
    """Plots a heatmap of 2D scalar data.

    :param img: 2D image data
    :param x: x-axis range for image data (min, max)
    :param y: y-axis range for image data (min, max)
    :param colormap: named color palette or Bokeh ColorMapper (see [Bokeh palettes](https://bokeh.pydata.org/en/latest/docs/reference/palettes.html))
    :param clim: color axis limits (min, max)
    :param clabel: color axis label
    :param title: figure title
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param xlim: x-axis limits (min, max)
    :param ylim: y-axis limits (min, max)
    :param width: figure width in pixels
    :param height: figure height in pixels
    :param hold: if set to True, output is not plotted immediately, but combined with the next plot

    >>> from arlpy.plot import image
    >>> import numpy as np
    >>> image(np.random.normal(size=(100,100)), colormap='Inferno256')
    """
    global _figure
    if x is None:
        x = [0, img.shape[1]-1]
    if y is None:
        y = [0, img.shape[0]-1]
    if xlim is None:
        xlim = x
    if ylim is None:
        ylim = y
    if _figure is None:
        _figure = _new_figure(title, width, height, xlabel, ylabel, xlim, ylim)
    if clim is None:
        clim = [_np.amin(img), _np.amax(img)]
    if isinstance(colormap, str):
        colormap = _bmodels.LinearColorMapper(palette=colormap, low=clim[0], high=clim[1])
    _figure.image([img], x=x[0], y=y[0], dw=x[-1]-x[0], dh=y[-1]-y[0], color_mapper=colormap)
    cbar = _bmodels.ColorBar(color_mapper=colormap, location=(0,0), title=clabel)
    _figure.add_layout(cbar, 'right')
    if not hold and not _hold:
        _bplt.show(_figure)
        _figure = None

def specgram(x, fs=2, nfft=None, noverlap=None, colormap='Plasma256', clim=None, clabel='dB', title=None, xlabel='Time (s)', ylabel='Frequency (Hz)', xlim=None, ylim=None, width=None, height=None, hold=False):
    """Plot spectrogram of a given time series signal.

    :param x: time series signal
    :param fs: sampling rate
    :param nfft: FFT size (see [scipy.signal.spectrogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html))
    :param noverlap: overlap size (see [scipy.signal.spectrogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html))
    :param colormap: named color palette or Bokeh ColorMapper (see [Bokeh palettes](https://bokeh.pydata.org/en/latest/docs/reference/palettes.html))
    :param clim: color axis limits (min, max), or dynamic range with respect to maximum
    :param clabel: color axis label
    :param title: figure title
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param xlim: x-axis limits (min, max)
    :param ylim: y-axis limits (min, max)
    :param width: figure width in pixels
    :param height: figure height in pixels
    :param hold: if set to True, output is not plotted immediately, but combined with the next plot

    >>> from arlpy.plot import specgram
    >>> import numpy as np
    >>> specgram(np.random.normal(size=(10000)), fs=10000, clim=30)
    """
    f, t, Sxx = _sig.spectrogram(x, fs=fs, nfft=nfft, noverlap=noverlap)
    Sxx = 10*_np.log10(Sxx)
    if isinstance(clim, float) or isinstance(clim, int):
        clim = [_np.max(Sxx)-clim, _np.max(Sxx)]
    image(Sxx, x=(t[0], t[-1]), y=(f[0], f[-1]), title=title, colormap=colormap, clim=clim, clabel=clabel, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, width=width, height=height, hold=hold)


def psd(x, fs=2, nfft=512, noverlap=None, window='hanning', title=None, xlabel='Frequency (Hz)', ylabel='Power spectral density (dB/Hz)', xlim=None, ylim=None, width=None, height=None, hold=False):
    """Plot power spectral density of a given time series signal.

    :param x: time series signal
    :param fs: sampling rate
    :param nfft: segment size (see [scipy.signal.welch](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html))
    :param noverlap: overlap size (see [scipy.signal.welch](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html))
    :param window: window to use (see [scipy.signal.welch](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html))
    :param title: figure title
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param xlim: x-axis limits (min, max)
    :param ylim: y-axis limits (min, max)
    :param width: figure width in pixels
    :param height: figure height in pixels
    :param hold: if set to True, output is not plotted immediately, but combined with the next plot

    >>> from arlpy.plot import psd
    >>> import numpy as np
    >>> psd(np.random.normal(size=(10000)), fs=10000)
    """
    f, Pxx = _sig.welch(x, fs=fs, nperseg=nfft, noverlap=noverlap, window=window)
    Pxx = 10*_np.log10(Pxx)
    if ylim is None:
        ylim = [_np.max(Pxx)-50, _np.max(Pxx)+10]
    plot(f, Pxx, title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, width=width, height=height, hold=hold)
