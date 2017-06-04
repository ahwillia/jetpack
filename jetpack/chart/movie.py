# -*- coding: utf-8 -*-
"""
Visualization tools for animations
"""
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy
import matplotlib.pyplot as plt
import numpy as np
from .utils import auto_subplot_layout

def imshow_videoclip(images, fps, subplots=None, clim=None, image_kw=dict(), **kwargs):

    # cast images to a numpy array
    images = np.array(images)
    if images.ndim > 4 or images.ndim < 3:
        raise ValueError('images must be a numpy array with either 3 dimension (single movie) or 4 dimensions (multiple movies).')
    elif images.ndim == 3:
        images = images[None, ...]
    
    # number of frames in video clip
    naxes = images.shape[0]
    nframes = images.shape[1]
    duration = nframes/fps

    # set up figure for plotting
    if subplots is None:
        subplots = auto_subplot_layout(naxes)
    fig, axes = plt.subplots(*subplots, **kwargs)
    axes = np.array([axes]) if isinstance(axes, plt.Axes) else axes
    axes = axes.ravel()
    
    # colormap limits
    clim = [(np.nanmin(images[a]), np.nanmax(images[a])) for a in range(naxes)]

    # wrap makeframe function with mpltfig_to_npimage
    def makeframe(t):
        i = int(nframes*t/duration)
        [axes[a].imshow(images[a,i], clim=clim[a], **image_kw) for a in range(naxes)]
        return mplfig_to_npimage(fig)
    
    # plot first frame
    [axes[a].imshow(images[a,0], clim=clim[a], **image_kw) for a in range(naxes)]
    fig.tight_layout()

    # return clip
    clip = mpy.VideoClip(makeframe, duration=duration)
    plt.close(fig)
    return clip
