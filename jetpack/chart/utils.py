# -*- coding: utf-8 -*-
"""
Utils
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)
import matplotlib.pyplot as plt
from functools import wraps
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.cm import get_cmap
import matplotlib

__all__ = ['setfontsize', 'noticks', 'nospines', 'noclips', 'breathe', 'setcolor',
           'get_bounds', 'tickdir', 'minimal_xticks', 'minimal_yticks',
           'categories_to_colors', 'simple_cmap', 'discretize_cmap']

def plotwrapper(fun):
    """
    Decorator that adds sane plotting defaults to the kwargs of a function
    """

    @wraps(fun)
    def wrapper(*args, **kwargs):

        if 'ax' not in kwargs:
            if 'fig' not in kwargs:
                kwargs['fig'] = plt.figure()
            kwargs['ax'] = kwargs['fig'].add_subplot(111)
        else:
            if 'fig' not in kwargs:
                kwargs['fig'] = kwargs['ax'].get_figure()

        fun(*args, **kwargs)
        return kwargs['ax']

    return wrapper


def axwrapper(fun):
    """
    Decorator that adds axis arguments, used for functions that modify
    and existing plot (this decorator will never create a new plot)
    """

    @wraps(fun)
    def wrapper(*args, **kwargs):
        if 'ax' not in kwargs:
            if 'fig' not in kwargs:
                kwargs['fig'] = plt.gcf()
            kwargs['ax'] = plt.gca()
        else:
            if 'fig' not in kwargs:
                kwargs['fig'] = kwargs['ax'].get_figure()
        fun(*args, **kwargs)
        return kwargs['ax']

    return wrapper


@plotwrapper
def setfontsize(size=18, **kwargs):
    """
    Sets the font size of the x- and y- tick labels of the current axes

    Parameters
    ----------
    size : int
        The font size to use

    """

    ax = kwargs['ax']
    ax.set_xticklabels(ax.get_xticks(), fontsize=size)
    ax.set_yticklabels(ax.get_yticks(), fontsize=size)

    return ax


@axwrapper
def noticks(**kwargs):
    """
    Clears tick marks (useful for images)
    """

    ax = kwargs['ax']
    ax.set_xticks([])
    ax.set_yticks([])


@axwrapper
def nospines(left=False, bottom=False, top=True, right=True, **kwargs):
    """
    Hides the specified axis spines (by default, right and top spines)
    """

    ax = kwargs['ax']

    # assemble args into dict
    disabled = dict(left=left, right=right, top=top, bottom=bottom)

    # disable spines
    for key in disabled:
        if disabled[key]:
            ax.spines[key].set_color('none')

    # disable xticks
    if disabled['top'] and disabled['bottom']:
        ax.set_xticks([])
    elif disabled['top']:
        ax.xaxis.set_ticks_position('bottom')
    elif disabled['bottom']:
        ax.xaxis.set_ticks_position('top')

    # disable yticks
    if disabled['left'] and disabled['right']:
        ax.set_yticks([])
    elif disabled['left']:
        ax.yaxis.set_ticks_position('right')
    elif disabled['right']:
        ax.yaxis.set_ticks_position('left')

    return ax

@axwrapper
def minimal_ticks(**kwargs):
    """
    Label only the first and last tick marks on x and y axes.
    """
    minimal_xticks(**kwargs)
    minimal_yticks(**kwargs)
    return kwargs['ax']

@axwrapper
def minimal_xticks(decimals=None, n_ticks=2, reset_bounds=True, **kwargs):
    """
    Label only the first and last tick marks on the x axis.
    """
    ax = kwargs['ax']

    ticks = ax.get_xticks()
    xl = ax.get_xlim()
    t0 = np.min(ticks[ticks > xl[0]])
    t1 = np.max(ticks[ticks < xl[1]])

    ax.set_xticks(np.linspace(t0, t1, n_ticks))

    if reset_bounds:
        ax.spines['top'].set_bounds(t0, t1)
        ax.spines['bottom'].set_bounds(t0, t1)

    return ax

@axwrapper
def minimal_yticks(decimals=None, n_ticks=2, reset_bounds=True, **kwargs):
    """
    Label only the first and last tick marks on the y axis.
    """
    ax = kwargs['ax']

    # get first and last tick
    ticks = ax.get_yticks()
    yl = ax.get_ylim()
    t0 = np.min(ticks[ticks > yl[0]])
    t1 = np.max(ticks[ticks < yl[1]])

    ax.set_yticks(np.linspace(t0, t1, n_ticks))

    if reset_bounds:
        ax.spines['right'].set_bounds(t0, t1)
        ax.spines['left'].set_bounds(t0, t1)
    
    return ax

def get_bounds(axis, ax=None):
    if ax is None:
        ax = plt.gca()

    axis_map = {
        "x": (ax.get_xticks, ax.get_xticklabels, ax.get_xlim, "bottom"),
        "y": (ax.get_yticks, ax.get_yticklabels, ax.get_ylim, "left"),
    }

    # get functions
    ticks, labels, limits, spine_key = axis_map[axis]

    if ax.spines[spine_key].get_bounds():
        return ax.spines[spine_key].get_bounds()
    else:
        lower, upper = None, None
        for tick, label in zip(ticks(), labels()):
            if label.get_text() != '':
                if lower is None:
                    lower = tick
                else:
                    upper = tick

        if lower is None or upper is None:
            return limits()

    return lower, upper

@axwrapper
def breathe(x_factor=0.05, y_factor=0.05, **kwargs):
    """
    Adds space between axes and plot
    """
    ax = kwargs['ax']

    if x_factor is not None:
        xa, xb = get_bounds('x', ax=ax)
        xrng = xb - xa
        ax.set_xlim(xa - x_factor * xrng, xb + x_factor * xrng)
        ax.spines['bottom'].set_bounds(xa, xb)
        ax.set_xticks([t for t in filter(lambda x: x >= xa and x <= xb, ax.xaxis.get_ticklocs())])

    if y_factor is not None:
        ya, yb = get_bounds('y', ax=ax)
        yrng = yb - ya
        ax.set_ylim(ya - y_factor * yrng, yb + y_factor * yrng)
        ax.spines['left'].set_bounds(ya, yb)
        ax.set_yticks([t for t in filter(lambda y: y >= ya and y <= yb, ax.yaxis.get_ticklocs())])

    nospines(**kwargs)

    return ax


@axwrapper
def tickdir(direction, **kwargs):
    ax = kwargs['ax']

    ax.xaxis.set_tick_params(direction=direction)
    ax.yaxis.set_tick_params(direction=direction)

    return ax


@axwrapper
def setcolor(color='#444444', **kwargs):
    ax = kwargs['ax']

    # set the tick parameters
    ax.tick_params(axis='x', colors=color)
    ax.tick_params(axis='y', colors=color)

    # set the label colors
    ax.xaxis.label.set_color(color)
    ax.yaxis.label.set_color(color)
    ax.set_xlabel(ax.get_xlabel(), color=color)
    ax.set_ylabel(ax.get_ylabel(), color=color)

    return ax

def noclips(fig):
    [o.set_clip_on(False) for o in fig.findobj()]
    return fig

def categories_to_colors(data, color_cycle=None):
    """Map categorical data to a color palette.

    Args
    ----
    data : array-like
        1-dimensional ndarray or list of categorical data
    color_cycle : array-like
        list of colors for each category in data

    Returns
    -------
    colors : list
        list mapping each datapoint to a color in color_cycle
    """
    if color_cycle is None:
        color_cycle = [(0.45, 0.62, 0.81),
                        (1.0, 0.62, 0.29),
                        (0.4, 0.75, 0.36),
                        (0.93, 0.4, 0.36),
                        (0.67, 0.55, 0.79),
                        (0.66, 0.47, 0.43),
                        (0.93, 0.59, 0.79),
                        (0.64, 0.64, 0.64),
                        (0.80, 0.8, 0.36),
                        (0.43, 0.8, 0.85)]
    elif not isinstance(color_cycle, list):
        try:
            color_cycle = list(color_cycle)
        except:
            raise ValueError('color_cycle should be specified as a list.')

    # list of unique values in the data vector
    categories = np.sort(np.array(list(set(data))))

    while len(categories) > len(color_cycle):
        color_cycle += color_cycle

    cat_to_col = {level: color for color, level in zip(color_cycle, categories)}
    return [cat_to_col[item] for item in data]

def simple_cmap(*colors, name='none'):
    """Create a colormap from a sequence of rgb values.

    cmap = simple_cmap((1,1,1), (1,0,0)) # white to red colormap
    cmap = simple_cmap('w', 'r')         # white to red colormap
    """

    # check inputs
    n_colors = len(colors)
    if n_colors <= 1:
        raise ValueError('Must specify at least two colors')

    # make sure colors are specified as rgb
    colors = [colorConverter.to_rgb(c) for c in colors]

    # set up colormap
    r, g, b = colors[0]
    cdict = {'red': [(0.0, r, r)], 'green': [(0.0, g, g)], 'blue': [(0.0, b, b)]}
    for i, (r, g, b) in enumerate(colors[1:]):
        idx = (i+1) / (n_colors-1)
        cdict['red'].append((idx, r, r))
        cdict['green'].append((idx, g, g))
        cdict['blue'].append((idx, b, b))

    return LinearSegmentedColormap(name, {k: tuple(v) for k, v in cdict.items()})

def discretize_cmap(cmap, n=100):
    """Sample a colormap at evenly spaced intervals

    colors = discretize_cmap('jet', 100)
    colors = discretize_cmap(matplotlib.cm.get_cmap('jet'), 100)
    """

    if isinstance(cmap, str):
        cmap = get_cmap(cmap)
    elif not isinstance(cmap, matplotlib.colors.LinearSegmentedColormap):
        raise ValueError('Colormap not recognized.')

    return [cmap(x) for x in np.linspace(0, 1, n)]

def auto_subplot_layout(n, tol=2.5):
    """Calculate roughly square layout for subplots
    """
    
    def _primes(n):
        """Computes the prime factorization for an integer
        """
        pfac = [] # prime factors
        d = 2
        while d*d <= n:
            while (n % d) == 0:
                pfac.append(d)
                n //= d
            d += 1
        if n > 1:
            pfac.append(n)
        return pfac
    
    def _isprime(n):
        """Returns whether an integer is prime or not.
        """
        return all([ n % d != 0 for d in range(2, n//2+1)])

    if not isinstance(n, int) or n <= 0:
        raise ValueError('number of subplots must be specified as a positive integer')

    if n == 1:
        return (1, 1)
    
    while _isprime(n) and n > 4:
        n = n+1

    p = _primes(n)

    # single row of plots
    if len(p) == 1:
        return 1, p[0]

    while len(p) > 2:
        if len(p) >= 4:
            p[1] = p[1]*p.pop()
            p[0] = p[0]*p.pop()
        else:
            # len(p) == 3
            p[0] = p[0]*p[1]
            p.pop(1)
        p = np.sort(p)

    if p[1]/p[0] > tol:
        return auto_subplot_layout(n+1, tol=tol)
    else:
        return p[0], p[1]
