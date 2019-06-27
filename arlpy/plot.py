##############################################################################
#
# Copyright (c) 2018, Mandar Chitre
#
# This file is part of arlpy which is released under Simplified BSD License.
# See file LICENSE or go to http://www.opensource.org/licenses/BSD-3-Clause
# for full license details.
#
##############################################################################

"""Easy-to-use plotting utilities based on `Bokeh <http://bokeh.pydata.org>`_."""

import numpy as _np
import os as _os
import warnings as _warnings
from tempfile import mkstemp as _mkstemp
import bokeh.plotting as _bplt
import bokeh.models as _bmodels
import bokeh.palettes as _bpal
import bokeh.resources as _bres
import bokeh.io as _bio
import scipy.signal as _sig

light_palette = ['mediumblue', 'crimson', 'forestgreen', 'gold', 'darkmagenta', 'olive', 'palevioletred', 'yellowgreen',
                 'deepskyblue', 'dimgray', 'indianred', 'mediumaquamarine', 'orange', 'saddlebrown', 'teal', 'mediumorchid']
dark_palette = ['lightskyblue', 'red', 'limegreen', 'salmon', 'magenta', 'forestgreen', 'silver', 'teal']

_figure = None
_figures = None
_hold = False
_figsize = (600, 400)
_color = 0
_notebook = False
_disable_js = False
_using_js = False
_interactive = True
_static_images = False
_colors = light_palette

try:
    get_ipython                     # check if we are using IPython
    _os.environ['JPY_PARENT_PID']   # and Jupyter
    _bplt.output_notebook(resources=_bres.INLINE, hide_banner=True)
    _notebook = True
except:
    pass                            # not in Jupyter, skip notebook initialization

def _new_figure(title, width, height, xlabel, ylabel, xlim, ylim, xtype, ytype, interactive):
    global _color
    if width is None:
        width = _figsize[0]
    if height is None:
        height = _figsize[1]
    _color = 0
    tools = []
    if interactive is None:
        interactive = _interactive
    if interactive:
        tools = 'pan,box_zoom,wheel_zoom,reset,save'
    f = _bplt.figure(title=title, plot_width=width, plot_height=height, x_range=xlim, y_range=ylim, x_axis_label=xlabel, y_axis_label=ylabel, x_axis_type=xtype, y_axis_type=ytype, tools=tools)
    f.toolbar.logo = None
    return f

def _process_canvas(figures):
    global _using_js
    if _disable_js:
        return
    if _using_js and len(figures) == 0:
        return
    disable = []
    i = 0
    for f in figures:
        i += 1
        if f is not None and f.tools == []:
            disable.append(i)
        else:
            pass
    if not _using_js and len(disable) == 0:
        return
    _using_js = True
    js = 'var disable = '+str(disable)
    js += """
    var clist = document.getElementsByClassName('bk-canvas');
    var j = 0;
    for (var i = 0; i < clist.length; i++) {
        if (clist[i].id == '') {
            j++;
            clist[i].id = 'bkc-'+String(i)+'-'+String(+new Date());
            if (disable.indexOf(j) >= 0) {
                var png = clist[i].toDataURL()
                var img = document.createElement('img')
                img.src = png
                clist[i].parentNode.replaceChild(img, clist[i])
            }
        }
    }
    """
    import IPython.display as _ipyd
    _ipyd.display(_ipyd.Javascript(js))

def _show_static_images(f):
    fh, fname = _mkstemp(suffix='.png')
    _os.close(fh)
    with _warnings.catch_warnings():      # to avoid displaying deprecation warning
        _warnings.simplefilter('ignore')  #   from bokeh 0.12.16
        _bio.export_png(f, fname)
    import IPython.display as _ipyd
    _ipyd.display(_ipyd.Image(filename=fname, embed=True))
    _os.unlink(fname)

def _show(f):
    if _figures is None:
        if _static_images:
            _show_static_images(f)
        else:
            _process_canvas([])
            _bplt.show(f)
            _process_canvas([f])
    else:
        _figures[-1].append(f)

def _hold_enable(enable):
    global _hold, _figure
    ohold = _hold
    _hold = enable
    if not _hold and _figure is not None:
        _show(_figure)
        _figure = None
    return ohold

def theme(name):
    """Set color theme.

    :param name: name of theme

    >>> import arlpy.plot
    >>> arlpy.plot.theme('dark')
    """
    if name == 'dark':
        name = 'dark_minimal'
        set_colors(dark_palette)
    elif name == 'light':
        name = 'light_minimal'
        set_colors(light_palette)
    _bio.curdoc().theme = name

def figsize(x, y):
    """Set the default figure size in pixels.

    :param x: figure width
    :param y: figure height
    """
    global _figsize
    _figsize = (x, y)

def interactive(b):
    """Set default interactivity for plots.

    :param b: True to enable interactivity, False to disable it
    """
    global _interactive
    _interactive = b

def enable_javascript(b):
    """Enable/disable Javascript.

    :param b: True to use Javacript, False to avoid use of Javascript

    Jupyterlab does not support Javascript output. To avoid error messages,
    Javascript can be disabled using this call. This removes an optimization
    to replace non-interactive plots with static images, but other than that
    does not affect functionality.
    """
    global _disable_js
    _disable_js = not b

def use_static_images(b=True):
    """Use static images instead of dynamic HTML/Javascript in Jupyter notebook.

    :param b: True to use static images, False to use HTML/Javascript

    Static images are useful when the notebook is to be exported as a markdown,
    LaTeX or PDF document, since dynamic HTML/Javascript is not rendered in these
    formats. When static images are used, all interactive functionality is disabled.

    To use static images, you must have the following packages installed:
    selenium, pillow, phantomjs.
    """
    global _static_images, _interactive
    if not b:
        _static_images = False
        return
    if not _notebook:
        _warnings.warn('Not running in a Jupyter notebook, static png support disabled')
        return
    _interactive = False
    _static_images = True

def hold(enable=True):
    """Combine multiple plots into one.

    :param enable: True to hold plot, False to release hold
    :returns: old state of hold if enable is True

    >>> import arlpy.plot
    >>> oh = arlpy.plot.hold()
    >>> arlpy.plot.plot([0,10], [0,10], color='blue', legend='A')
    >>> arlpy.plot.plot([10,0], [0,10], marker='o', color='green', legend='B')
    >>> arlpy.plot.hold(oh)
    """
    rv = _hold_enable(enable)
    return rv if enable else None

class figure:
    """Create a new figure, and optionally automatically display it.

    :param title: figure title
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param xlim: x-axis limits (min, max)
    :param ylim: y-axis limits (min, max)
    :param xtype: x-axis type ('auto', 'linear', 'log', etc)
    :param ytype: y-axis type ('auto', 'linear', 'log', etc)
    :param width: figure width in pixels
    :param height: figure height in pixels
    :param interactive: enable interactive tools (pan, zoom, etc) for plot

    This function can be used in standalone mode to create a figure:

    >>> import arlpy.plot
    >>> arlpy.plot.figure(title='Demo 1', width=500)
    >>> arlpy.plot.plot([0,10], [0,10])

    Or it can be used as a context manager to create, hold and display a figure:

    >>> import arlpy.plot
    >>> with arlpy.plot.figure(title='Demo 2', width=500):
    >>>     arlpy.plot.plot([0,10], [0,10], color='blue', legend='A')
    >>>     arlpy.plot.plot([10,0], [0,10], marker='o', color='green', legend='B')

    It can even be used as a context manager to work with Bokeh functions directly:

    >>> import arlpy.plot
    >>> with arlpy.plot.figure(title='Demo 3', width=500) as f:
    >>>     f.line([0,10], [0,10], line_color='blue')
    >>>     f.square([3,7], [4,5], line_color='green', fill_color='yellow', size=10)
    """

    def __init__(self, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, xtype='auto', ytype='auto', width=None, height=None, interactive=None):
        global _figure
        _figure = _new_figure(title, width, height, xlabel, ylabel, xlim, ylim, xtype, ytype, interactive)

    def __enter__(self):
        global _hold
        _hold = True
        return _figure

    def __exit__(self, *args):
        global _hold, _figure
        _hold = False
        _show(_figure)
        _figure = None

class many_figures:
    """Create a grid of many figures.

    :param figsize: default size of figure in grid as (width, height)

    >>> import arlpy.plot
    >>> with arlpy.plot.many_figures(figsize=(300,200)):
    >>>     arlpy.plot.plot([0,10], [0,10])
    >>>     arlpy.plot.plot([0,10], [0,10])
    >>>     arlpy.plot.next_row()
    >>>     arlpy.plot.next_column()
    >>>     arlpy.plot.plot([0,10], [0,10])
    """

    def __init__(self, figsize=None):
        self.figsize = figsize

    def __enter__(self):
        global _figures, _figsize
        _figures = [[]]
        self.ofigsize = _figsize
        if self.figsize is not None:
            _figsize = self.figsize

    def __exit__(self, *args):
        global _figures, _figsize
        if len(_figures) > 1 or len(_figures[0]) > 0:
            f = _bplt.gridplot(_figures, merge_tools=False)
            if _static_images:
                _show_static_images(f)
            else:
                _process_canvas([])
                _bplt.show(f)
                _process_canvas([item for sublist in _figures for item in sublist])
        _figures = None
        _figsize = self.ofigsize

def next_row():
    """Move to the next row in a grid of many figures."""
    global _figures
    if _figures is not None:
        _figures.append([])

def next_column():
    """Move to the next column in a grid of many figures."""
    global _figures
    if _figures is not None:
        _figures[-1].append(None)

def gcf():
    """Get the current figure.

    :returns: handle to the current figure
    """
    return _figure

def plot(x, y=None, fs=None, maxpts=10000, pooling=None, color=None, style='solid', thickness=1, marker=None, filled=False, size=6, mskip=0, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, xtype='auto', ytype='auto', width=None, height=None, legend=None, hold=False, interactive=None):
    """Plot a line graph or time series.

    :param x: x data or time series data (if y is None)
    :param y: y data or None (if time series)
    :param fs: sampling rate for time series data
    :param maxpts: maximum number of points to plot (downsampled if more points provided)
    :param pooling: pooling for downsampling (None, 'max', 'min', 'mean', 'median')
    :param color: line color (see `Bokeh colors <https://bokeh.pydata.org/en/latest/docs/reference/colors.html>`_)
    :param style: line style ('solid', 'dashed', 'dotted', 'dotdash', 'dashdot', None)
    :param thickness: line width in pixels
    :param marker: point markers ('.', 'o', 's', '*', 'x', '+', 'd', '^')
    :param filled: filled markers or outlined ones
    :param size: marker size
    :param mskip: number of points to skip marking (to avoid too many markers)
    :param title: figure title
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param xlim: x-axis limits (min, max)
    :param ylim: y-axis limits (min, max)
    :param xtype: x-axis type ('auto', 'linear', 'log', etc)
    :param ytype: y-axis type ('auto', 'linear', 'log', etc)
    :param width: figure width in pixels
    :param height: figure height in pixels
    :param legend: legend text
    :param interactive: enable interactive tools (pan, zoom, etc) for plot
    :param hold: if set to True, output is not plotted immediately, but combined with the next plot

    >>> import arlpy.plot
    >>> import numpy as np
    >>> arlpy.plot.plot([0,10], [1,-1], color='blue', marker='o', filled=True, legend='A', hold=True)
    >>> arlpy.plot.plot(np.random.normal(size=1000), fs=100, color='green', legend='B')
    """
    global _figure, _color
    x = _np.array(x, ndmin=1, dtype=_np.float, copy=False)
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
        y = _np.array(y, ndmin=1, dtype=_np.float, copy=False)
    if _figure is None:
        _figure = _new_figure(title, width, height, xlabel, ylabel, xlim, ylim, xtype, ytype, interactive)
    if color is None:
        color = _colors[_color % len(_colors)]
        _color += 1
    if x.size > maxpts:
        n = int(_np.ceil(x.size/maxpts))
        x = x[::n]
        desc = 'Downsampled by '+str(n)
        if pooling is None:
            y = y[::n]
        elif pooling == 'max':
            desc += ', '+pooling+' pooled'
            y = _np.amax(_np.reshape(y[:n*(y.size//n)], (-1, n)), axis=1)
        elif pooling == 'min':
            desc += ', '+pooling+' pooled'
            y = _np.amin(_np.reshape(y[:n*(y.size//n)], (-1, n)), axis=1)
        elif pooling == 'mean':
            desc += ', '+pooling+' pooled'
            y = _np.mean(_np.reshape(y[:n*(y.size//n)], (-1, n)), axis=1)
        elif pooling == 'median':
            desc += ', '+pooling+' pooled'
            y = _np.mean(_np.reshape(y[:n*(y.size//n)], (-1, n)), axis=1)
        else:
            _warnings.warn('Unknown pooling: '+pooling)
            y = y[::n]
        if len(x) > len(y):
            x = x[:len(y)]
        _figure.add_layout(_bmodels.Label(x=5, y=5, x_units='screen', y_units='screen', text=desc, text_font_size="8pt", text_alpha=0.5))
    if style is not None:
        _figure.line(x, y, line_color=color, line_dash=style, line_width=thickness, legend=legend)
    if marker is not None:
        scatter(x[::(mskip+1)], y[::(mskip+1)], marker=marker, filled=filled, size=size, color=color, legend=legend, hold=True)
    if not hold and not _hold:
        _show(_figure)
        _figure = None

def scatter(x, y, marker='.', filled=False, size=6, color=None, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, xtype='auto', ytype='auto', width=None, height=None, legend=None, hold=False, interactive=None):
    """Plot a scatter plot.

    :param x: x data
    :param y: y data
    :param color: marker color (see `Bokeh colors`_)
    :param marker: point markers ('.', 'o', 's', '*', 'x', '+', 'd', '^')
    :param filled: filled markers or outlined ones
    :param size: marker size
    :param title: figure title
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param xlim: x-axis limits (min, max)
    :param ylim: y-axis limits (min, max)
    :param xtype: x-axis type ('auto', 'linear', 'log', etc)
    :param ytype: y-axis type ('auto', 'linear', 'log', etc)
    :param width: figure width in pixels
    :param height: figure height in pixels
    :param legend: legend text
    :param interactive: enable interactive tools (pan, zoom, etc) for plot
    :param hold: if set to True, output is not plotted immediately, but combined with the next plot

    >>> import arlpy.plot
    >>> import numpy as np
    >>> arlpy.plot.scatter(np.random.normal(size=100), np.random.normal(size=100), color='blue', marker='o')
    """
    global _figure, _color
    if _figure is None:
        _figure = _new_figure(title, width, height, xlabel, ylabel, xlim, ylim, xtype, ytype, interactive)
    x = _np.array(x, ndmin=1, dtype=_np.float, copy=False)
    y = _np.array(y, ndmin=1, dtype=_np.float, copy=False)
    if color is None:
        color = _colors[_color % len(_colors)]
        _color += 1
    if marker == '.':
        _figure.circle(x, y, size=size/2, line_color=color, fill_color=color, legend=legend)
    elif marker == 'o':
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
        _warnings.warn('Bad marker type: '+marker)
    if not hold and not _hold:
        _show(_figure)
        _figure = None

def image(img, x=None, y=None, colormap='Plasma256', clim=None, clabel=None, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, xtype='auto', ytype='auto', width=None, height=None, hold=False, interactive=None):
    """Plot a heatmap of 2D scalar data.

    :param img: 2D image data
    :param x: x-axis range for image data (min, max)
    :param y: y-axis range for image data (min, max)
    :param colormap: named color palette or Bokeh ColorMapper (see `Bokeh palettes <https://bokeh.pydata.org/en/latest/docs/reference/palettes.html>`_)
    :param clim: color axis limits (min, max)
    :param clabel: color axis label
    :param title: figure title
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param xlim: x-axis limits (min, max)
    :param ylim: y-axis limits (min, max)
    :param xtype: x-axis type ('auto', 'linear', 'log', etc)
    :param ytype: y-axis type ('auto', 'linear', 'log', etc)
    :param width: figure width in pixels
    :param height: figure height in pixels
    :param interactive: enable interactive tools (pan, zoom, etc) for plot
    :param hold: if set to True, output is not plotted immediately, but combined with the next plot

    >>> import arlpy.plot
    >>> import numpy as np
    >>> arlpy.plot.image(np.random.normal(size=(100,100)), colormap='Inferno256')
    """
    global _figure
    if x is None:
        x = (0, img.shape[1]-1)
    if y is None:
        y = (0, img.shape[0]-1)
    if xlim is None:
        xlim = x
    if ylim is None:
        ylim = y
    if _figure is None:
        _figure = _new_figure(title, width, height, xlabel, ylabel, xlim, ylim, xtype, ytype, interactive)
    if clim is None:
        clim = [_np.amin(img), _np.amax(img)]
    if not isinstance(colormap, _bmodels.ColorMapper):
        colormap = _bmodels.LinearColorMapper(palette=colormap, low=clim[0], high=clim[1])
    _figure.image([img], x=x[0], y=y[0], dw=x[-1]-x[0], dh=y[-1]-y[0], color_mapper=colormap)
    cbar = _bmodels.ColorBar(color_mapper=colormap, location=(0,0), title=clabel)
    _figure.add_layout(cbar, 'right')
    if not hold and not _hold:
        _show(_figure)
        _figure = None

def vlines(x, color='gray', style='dashed', thickness=1, hold=False):
    """Draw vertical lines on a plot.

    :param x: x location of lines
    :param color: line color (see `Bokeh colors`_)
    :param style: line style ('solid', 'dashed', 'dotted', 'dotdash', 'dashdot')
    :param thickness: line width in pixels
    :param hold: if set to True, output is not plotted immediately, but combined with the next plot

    >>> import arlpy.plot
    >>> arlpy.plot.plot([0, 20], [0, 10], hold=True)
    >>> arlpy.plot.vlines([7, 12])
    """
    global _figure
    if _figure is None:
        return
    x = _np.array(x, ndmin=1, dtype=_np.float, copy=False)
    for j in range(x.size):
        _figure.add_layout(_bmodels.Span(location=x[j], dimension='height', line_color=color, line_dash=style, line_width=thickness))
    if not hold and not _hold:
        _show(_figure)
        _figure = None

def hlines(y, color='gray', style='dashed', thickness=1, hold=False):
    """Draw horizontal lines on a plot.

    :param y: y location of lines
    :param color: line color (see `Bokeh colors`_)
    :param style: line style ('solid', 'dashed', 'dotted', 'dotdash', 'dashdot')
    :param thickness: line width in pixels
    :param hold: if set to True, output is not plotted immediately, but combined with the next plot

    >>> import arlpy.plot
    >>> arlpy.plot.plot([0, 20], [0, 10], hold=True)
    >>> arlpy.plot.hlines(3, color='red', style='dotted')
    """
    global _figure
    if _figure is None:
        return
    y = _np.array(y, ndmin=1, dtype=_np.float, copy=False)
    for j in range(y.size):
        _figure.add_layout(_bmodels.Span(location=y[j], dimension='width', line_color=color, line_dash=style, line_width=thickness))
    if not hold and not _hold:
        _show(_figure)
        _figure = None

def text(x, y, s, color='gray', size='8pt', hold=False):
    """Add text annotation to a plot.

    :param x: x location of left of text
    :param y: y location of bottom of text
    :param s: text to add
    :param color: text color (see `Bokeh colors`_)
    :param size: text size (e.g. '12pt', '3em')
    :param hold: if set to True, output is not plotted immediately, but combined with the next plot

    >>> import arlpy.plot
    >>> arlpy.plot.plot([0, 20], [0, 10], hold=True)
    >>> arlpy.plot.text(7, 3, 'demo', color='orange')
    """
    global _figure
    if _figure is None:
        return
    _figure.add_layout(_bmodels.Label(x=x, y=y, text=s, text_font_size=size, text_color=color))
    if not hold and not _hold:
        _show(_figure)
        _figure = None

def box(left=None, right=None, top=None, bottom=None, color='yellow', alpha=0.1, hold=False):
    """Add a highlight box to a plot.

    :param left: x location of left of box
    :param right: x location of right of box
    :param top: y location of top of box
    :param bottom: y location of bottom of box
    :param color: text color (see `Bokeh colors`_)
    :param alpha: transparency (0-1)
    :param hold: if set to True, output is not plotted immediately, but combined with the next plot

    >>> import arlpy.plot
    >>> arlpy.plot.plot([0, 20], [0, 10], hold=True)
    >>> arlpy.plot.box(left=5, right=10, top=8)
    """
    global _figure
    if _figure is None:
        return
    _figure.add_layout(_bmodels.BoxAnnotation(left=left, right=right, top=top, bottom=bottom, fill_color=color, fill_alpha=alpha))
    if not hold and not _hold:
        _show(_figure)
        _figure = None

def color(n):
    """Get a numbered color to cycle over a set of colors.

    >>> import arlpy.plot
    >>> arlpy.plot.color(0)
    'blue'
    >>> arlpy.plot.color(1)
    'red'
    >>> arlpy.plot.plot([0, 20], [0, 10], color=arlpy.plot.color(3))
    """
    return _colors[n % len(_colors)]

def set_colors(c):
    """Provide a list of named colors to cycle over.

    >>> import arlpy.plot
    >>> arlpy.plot.set_colors(['red', 'blue', 'green', 'black'])
    >>> arlpy.plot.color(2)
    'green'
    """
    global _colors
    _colors = c

def specgram(x, fs=2, nfft=None, noverlap=None, colormap='Plasma256', clim=None, clabel='dB', title=None, xlabel='Time (s)', ylabel='Frequency (Hz)', xlim=None, ylim=None, width=None, height=None, hold=False, interactive=None):
    """Plot spectrogram of a given time series signal.

    :param x: time series signal
    :param fs: sampling rate
    :param nfft: FFT size (see `scipy.signal.spectrogram <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html>`_)
    :param noverlap: overlap size (see `scipy.signal.spectrogram`_)
    :param colormap: named color palette or Bokeh ColorMapper (see `Bokeh palettes`_)
    :param clim: color axis limits (min, max), or dynamic range with respect to maximum
    :param clabel: color axis label
    :param title: figure title
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param xlim: x-axis limits (min, max)
    :param ylim: y-axis limits (min, max)
    :param width: figure width in pixels
    :param height: figure height in pixels
    :param interactive: enable interactive tools (pan, zoom, etc) for plot
    :param hold: if set to True, output is not plotted immediately, but combined with the next plot

    >>> import arlpy.plot
    >>> import numpy as np
    >>> arlpy.plot.specgram(np.random.normal(size=(10000)), fs=10000, clim=30)
    """
    f, t, Sxx = _sig.spectrogram(x, fs=fs, nperseg=nfft, noverlap=noverlap)
    Sxx = 10*_np.log10(Sxx+_np.finfo(float).eps)
    if isinstance(clim, float) or isinstance(clim, int):
        clim = (_np.max(Sxx)-clim, _np.max(Sxx))
    image(Sxx, x=(t[0], t[-1]), y=(f[0], f[-1]), title=title, colormap=colormap, clim=clim, clabel=clabel, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, width=width, height=height, hold=hold, interactive=interactive)

def psd(x, fs=2, nfft=512, noverlap=None, window='hanning', color=None, style='solid', thickness=1, marker=None, filled=False, size=6, title=None, xlabel='Frequency (Hz)', ylabel='Power spectral density (dB/Hz)', xlim=None, ylim=None, width=None, height=None, legend=None, hold=False, interactive=None):
    """Plot power spectral density of a given time series signal.

    :param x: time series signal
    :param fs: sampling rate
    :param nfft: segment size (see `scipy.signal.welch <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html>`_)
    :param noverlap: overlap size (see `scipy.signal.welch`_)
    :param window: window to use (see `scipy.signal.welch`_)
    :param color: line color (see `Bokeh colors`_)
    :param style: line style ('solid', 'dashed', 'dotted', 'dotdash', 'dashdot')
    :param thickness: line width in pixels
    :param marker: point markers ('.', 'o', 's', '*', 'x', '+', 'd', '^')
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
    :param interactive: enable interactive tools (pan, zoom, etc) for plot
    :param hold: if set to True, output is not plotted immediately, but combined with the next plot

    >>> import arlpy.plot
    >>> import numpy as np
    >>> arlpy.plot.psd(np.random.normal(size=(10000)), fs=10000)
    """
    f, Pxx = _sig.welch(x, fs=fs, nperseg=nfft, noverlap=noverlap, window=window)
    Pxx = 10*_np.log10(Pxx+_np.finfo(float).eps)
    if xlim is None:
        xlim = (0, fs/2)
    if ylim is None:
        ylim = (_np.max(Pxx)-50, _np.max(Pxx)+10)
    plot(f, Pxx, color=color, style=style, thickness=thickness, marker=marker, filled=filled, size=size, title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, maxpts=len(f), width=width, height=height, hold=hold, legend=legend, interactive=interactive)

def iqplot(data, marker='.', color=None, labels=None, filled=False, size=None, title=None, xlabel=None, ylabel=None, xlim=[-2, 2], ylim=[-2, 2], width=None, height=None, hold=False, interactive=None):
    """Plot signal points.

    :param data: complex baseband signal points
    :param marker: point markers ('.', 'o', 's', '*', 'x', '+', 'd', '^')
    :param color: marker/text color (see `Bokeh colors`_)
    :param labels: label for each signal point, or True to auto-generate labels
    :param filled: filled markers or outlined ones
    :param size: marker/text size (e.g. 5, '8pt')
    :param title: figure title
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param xlim: x-axis limits (min, max)
    :param ylim: y-axis limits (min, max)
    :param width: figure width in pixels
    :param height: figure height in pixels
    :param interactive: enable interactive tools (pan, zoom, etc) for plot
    :param hold: if set to True, output is not plotted immediately, but combined with the next plot

    >>> import arlpy
    >>> import arlpy.plot
    >>> arlpy.plot.iqplot(arlpy.comms.psk(8))
    >>> arlpy.plot.iqplot(arlpy.comms.qam(16), color='red', marker='x')
    >>> arlpy.plot.iqplot(arlpy.comms.psk(4), labels=['00', '01', '11', '10'])
    """
    data = _np.asarray(data, dtype=_np.complex)
    if not _hold:
        figure(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, width=width, height=height, interactive=interactive)
    if labels is None:
        if size is None:
            size = 5
        scatter(data.real, data.imag, marker=marker, filled=filled, color=color, size=size, hold=hold)
    else:
        if labels == True:
            labels = range(len(data))
        if color is None:
            color = 'black'
        plot([0], [0], hold=True)
        for i in range(len(data)):
            text(data[i].real, data[i].imag, str(labels[i]), color=color, size=size, hold=True if i < len(data)-1 else hold)

def freqz(b, a=1, fs=2.0, worN=None, whole=False, degrees=True, style='solid', thickness=1, title=None, xlabel='Frequency (Hz)', xlim=None, ylim=None, width=None, height=None, hold=False, interactive=None):
    """Plot frequency response of a filter.

    This is a convenience function to plot frequency response, and internally uses
    :func:`scipy.signal.freqz` to estimate the response. For further details, see the
    documentation for :func:`scipy.signal.freqz`.

    :param b: numerator of a linear filter
    :param a: denominator of a linear filter
    :param fs: sampling rate in Hz (optional, normalized frequency if not specified)
    :param worN: see :func:`scipy.signal.freqz`
    :param whole: see :func:`scipy.signal.freqz`
    :param degrees: True to display phase in degrees, False for radians
    :param style: line style ('solid', 'dashed', 'dotted', 'dotdash', 'dashdot')
    :param thickness: line width in pixels
    :param title: figure title
    :param xlabel: x-axis label
    :param ylabel1: y-axis label for magnitude
    :param ylabel2: y-axis label for phase
    :param xlim: x-axis limits (min, max)
    :param ylim: y-axis limits (min, max)
    :param width: figure width in pixels
    :param height: figure height in pixels
    :param interactive: enable interactive tools (pan, zoom, etc) for plot
    :param hold: if set to True, output is not plotted immediately, but combined with the next plot

    >>> import arlpy
    >>> arlpy.plot.freqz([1,1,1,1,1], fs=120000);
    """
    w, h = _sig.freqz(b, a, worN, whole)
    Hxx = 20*_np.log10(abs(h)+_np.finfo(float).eps)
    f = w*fs/(2*_np.pi)
    if xlim is None:
        xlim = (0, fs/2)
    if ylim is None:
        ylim = (_np.max(Hxx)-50, _np.max(Hxx)+10)
    figure(title=title, xlabel=xlabel, ylabel='Amplitude (dB)', xlim=xlim, ylim=ylim, width=width, height=height, interactive=interactive)
    _hold_enable(True)
    plot(f, Hxx, color=color(0), style=style, thickness=thickness, legend='Magnitude')
    fig = gcf()
    units = 180/_np.pi if degrees else 1
    fig.extra_y_ranges = {'phase': _bmodels.Range1d(start=-_np.pi*units, end=_np.pi*units)}
    fig.add_layout(_bmodels.LinearAxis(y_range_name='phase', axis_label='Phase (degrees)' if degrees else 'Phase (radians)'), 'right')
    phase = _np.angle(h)*units
    fig.line(f, phase, line_color=color(1), line_dash=style, line_width=thickness, legend='Phase', y_range_name='phase')
    _hold_enable(hold)
