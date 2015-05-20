'''
Code to create visualizations for "From manufacturing belt, to rust belt, to
college country - A Visual Narrative of the US Urban Growth", by Dani
Arribas-Bel and Michiel Gerritse

Author: Dani Arribas-Bel <D.Arribas-Bel@bham.ac.uk>
...

Copyright (c) 2014, Daniel Arribas-Bel

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* The name of Daniel Arribas-Bel may not be used to endorse or promote products
  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''

import time
import pysal as ps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pysal as ps
from math import acos, degrees
from pyGDsandbox.dataIO import dbf2df, cols_as_mi
from pysal.contrib.viz import mapping as viz
from matplotlib.colors import colorConverter as cc
from matplotlib.collections import LineCollection

vars = ['k', 'service', 'manufacturing', 'valheating', 'valmaxtemp', \
        'valmeantemp', 'college']

direc_bg = 'w'

def auto_fig(coefs, method, var='manufacturing', folder_out=None):
    '''
    Automatically pick-up all the transitions with data for `var`
    and push them to `var_transitions`
    ...

    NOTE: - the figure can get very large, better to run on the notebook
          - method can be `var_maps` or `var_transitions`
    '''
    lims = None
    db = coefs[['year', 'we'+var, 're'+var]]
    years2i = []
    dists = []
    years = coefs.year.unique()
    years.sort()
    for i in range(years.shape[0]-1):
        t = coefs[coefs['year'] == years[i]].set_index('pid')
        tp1 = coefs[coefs['year'] == years[i+1]].set_index('pid')
        vecs = t.join(tp1, lsuffix='_t', rsuffix='_tp1')\
            [['we'+var+'_t', 're'+var+'_t', 'we'+var+'_tp1', 're'+var+'_tp1']]\
            .dropna()
        if vecs.shape[0] > 1:
            xlimax = np.abs(vecs['re'+var+'_tp1'] - vecs['re'+var+'_t']).max()
            ylimax = np.abs(vecs['we'+var+'_tp1'] - vecs['we'+var+'_t']).max()
            max_dist = np.sqrt((xlimax)**2 + (ylimax)**2)
            dists.append(max_dist)
            years2i.append(years[i])
            years2i.append(years[i+1])
    maxd = np.max(dists)
    lims = ((-maxd, maxd), (-maxd, maxd))
    years2i = list(set(years2i))
    years2i.sort()
    _ = method(coefs, years2i, var, lims=lims, folder_out=folder_out)
    return None

def var_maps(coefs, years, var='manufacturing', lims=None):
    '''
    Plot all the transition maps over the sequence of `years` for a variable in a
    single figure
    '''
    y = len(years)
    # Get scalers for rents and wages
    max1s, max2s = [], []
    for i in range(y-1):
        src = years[i]
        tgt = years[i+1]
        colors1, alphas1 = trans_map(coefs, src, tgt, 're', var)
        colors2, alphas2 = trans_map(coefs, src, tgt, 'we', var)
        max1s.append(np.max(alphas1))
        max2s.append(np.max(alphas2))
    max1 = np.max(max1s)
    max2 = np.max(max2s)
    # Plot
    f = plt.figure(figsize=(14, 6*(y-1)))
    dotsize = 60

    for i in range(y-1):
        src = years[i]
        tgt = years[i+1]

        colors1, alphas1 = trans_map(coefs, src, tgt, 're', var)
        colors2, alphas2 = trans_map(coefs, src, tgt, 'we', var)
        # Standardize by the (sqrt) re/we max of the period
        alphas1 = ( np.sqrt(alphas1 * 1.) / np.sqrt(max1) ).values
        alphas2 = ( np.sqrt(alphas2 * 1.) / np.sqrt(max2) ).values

        ax1 = f.add_subplot(y-1, 2, (i*2)+1)
        _ = choro(colors1, alphas1, ax1, s=dotsize)
        ax1.set_title('Rent effects %i-%i'%(src, tgt))

        ax2 = f.add_subplot(y-1, 2, (i*2)+2)
        _ = choro(colors2, alphas2, ax2, s=dotsize)
        ax2.set_title('Wage effects %i-%i'%(src, tgt))
    f.suptitle(var)
    plt.show()
    return None

def trans_map(coefs, src=2000, tgt=2010, effect='re', var='manufacturing'):
    cm = mpl.cm.Set1
    t = coefs[coefs['year'] == src].set_index('pid')
    tp1 = coefs[coefs['year'] == tgt].set_index('pid')
    chgs = t.join(tp1, lsuffix='_t', rsuffix='_tp1')\
        [[effect+var+'_t', effect+var+'_tp1']]\
        .dropna()\
        .rename(columns={effect+var+'_t': 't', \
                        effect+var+'_tp1': 'tp1'})
    chgs['chg'] = chgs['tp1'] - chgs['t']
    chgs['pos'] = chgs['chg'] > 0
    chgs['mag'] = np.abs(chgs['chg'])
    chgs['color'] = None
    # Blues if increasing
    blues = chgs['pos']==True
    chgs.ix[blues, 'color'] = chgs.ix[blues, 'mag']\
            .apply(lambda x: (0., 0., 1., 1))
    #       .apply(mpl.cm.Blues)
    # Reds if decreasing
    chgs.ix[chgs['pos']==False, 'color'] = chgs.ix[chgs['pos']==False, 'mag']\
            .apply(lambda x: (1., 0., 0., 1))
    #       .apply(mpl.cm.Reds)
    # Switch back to 'metarea' to conform with other functions
    chgs = chgs.rename(coefs.set_index('pid')['metarea'].drop_duplicates())
    return chgs['color'], chgs['mag']

def var_transitions(coefs, years, var='manufacturing', lims=None, scaler=None):
    '''
    Plot all the transitions over the sequence of `years` for variable `var`
    using `lims` as a scale in a single figure
    '''
    y = len(years)
    f = plt.figure(figsize=(14, 6*(y-1)))
    for i in range(y-1):
        src = years[i]
        tgt = years[i+1]
        dax = f.add_subplot(y-1, 2, (i*2)+1)
        _, colors, alphas = directions(coefs, src, tgt, var, ax=dax)
        if lims:
            xlim, ylim = lims
            dax.set_xlim(xlim)
            dax.set_ylim(ylim)
        max = f.add_subplot(y-1, 2, (i*2)+2)
        _ = choro(colors, alphas, max)
    plt.show()
    return None

def var_transition(coefs, years, var='manufacturing', lims=None, scaler=None):
    '''
    `var_transitions` with only directional plot
    '''
    y = len(years)
    f = plt.figure(figsize=(7, 6*(y-1)))
    for i in range(y-1):
        src = years[i]
        tgt = years[i+1]
        dax = f.add_subplot(y-1, 1, i+1)
        _, colors, alphas = directions(coefs, src, tgt, var, ax=dax)
        if lims:
            xlim, ylim = lims
            dax.set_xlim(xlim)
            dax.set_ylim(ylim)
    plt.show()
    return None

def var_transition_sf(coefs, years, var='manufacturing', lims=None,
        scaler=None, folder_out=None):
    '''
    `var_transitions` with only directional plot. Single figures.
    '''
    y = len(years)
    for i in range(y-1):
        src = years[i]
        tgt = years[i+1]
        f = plt.figure(figsize=(6, 6))
        dax = f.add_subplot(111)
        _, colors, alphas = directions(coefs, src, tgt, var, ax=dax,
                monochrome=True)
        if lims:
            xlim, ylim = lims
            dax.set_xlim(xlim)
            dax.set_ylim(ylim)
            dax = _add_text(dax, xlim[1])
        if folder_out:
            saveto = folder_out + var + '_' + str(src) + '-' + str(tgt) + '.png'
            plt.savefig(saveto)
        plt.show()
    return None

def directions(coefs, src=2000, tgt=2010, var='manufacturing', ax=None,
        indexer='metarea', monochrome=False):
    '''
    Plot with transitions zeroed-in to origin
    '''
    cm = mpl.cm.Set1
    f = None
    if not ax:
        f, ax = plt.subplots(1)
    t = coefs[coefs['year'] == src].set_index(indexer)
    tp1 = coefs[coefs['year'] == tgt].set_index(indexer)
    vecs = t.join(tp1, lsuffix='_t', rsuffix='_tp1')\
        [['we'+var+'_t', 're'+var+'_t', 'we'+var+'_tp1', 're'+var+'_tp1']]\
        .dropna()
    colors = None
    alphas = None
    if vecs.shape[0] > 0:
        vecsZ = vecs[['we'+var+'_tp1']].sub(vecs['we'+var+'_t'], axis=0)
        vecsZ['re'+var+'_tp1'] = vecs['re'+var+'_tp1'] - vecs['re'+var+'_t']
        vecsZ['we'+var+'_t'] = 0
        vecsZ['re'+var+'_t'] = 0
        vecsZL = vecsZ.apply(_liner, axis=1, var=var).tolist()
        x1 = vecsZ['re'+var+'_tp1']
        y1 = vecsZ['we'+var+'_tp1']
        max_dist = np.sqrt((x1)**2 + (y1)**2).max()#Dists from origin (0, 0)
        xy1 = np.hstack((x1[:, None], y1[:, None]))
        lengths = _length(xy1)
        alphas = lengths * 1. / lengths.max()
        angles = np.apply_along_axis(_angler, 1, xy1)
        c = map(cm, angles / 360.)
        colors = pd.Series(c, index=vecsZ.index)
        if monochrome:
            lines = LineCollection(vecsZL, color='k', alpha=0.5)
        else:
            lines = LineCollection(vecsZL, color=c)
        ax.add_collection(lines)
        ax.set_xlim((-max_dist, max_dist))
        ax.set_ylim((-max_dist, max_dist))
        backcolor = '0.6'
        ax.set_xticklabels('')
        ax.set_xticks([])
        ax.set_yticklabels('')
        ax.set_yticks([])
        #ax.set_xlabel("<-- Disamenity | Amenity -->", size=25, color=backcolor)
        #ax.set_ylabel("<-- Cons. | Prod. -->", size=25, color=backcolor)
        ax.set_title("%i - %i"%(src, tgt), color=backcolor, size=30)
        ax.set_axis_bgcolor(direc_bg)
        plt.setp(ax.spines.values(), color=backcolor)
        # Upper right
        max_dist = ax.get_xlim()[1]
    if f:
        plt.show()
    return ax, colors, alphas

def _add_text(ax, max_dist):
    bg_alpha = 0.4
    backcolor = '0.6'
    ax.fill([0, max_dist, max_dist, 0], \
            [0, 0, max_dist, max_dist], \
            color=quad_codes['Production amenity'], \
            ec='none', alpha=bg_alpha)
    plt.text(max_dist * 0.5, max_dist * 0.75,  \
            'Production \namenity', \
            color=backcolor, fontsize=25, \
            horizontalalignment='center', verticalalignment='center')
    # Upper left
    plt.text(-max_dist * 0.5, max_dist * 0.75,  \
            'Consumption \ndisamenity', \
            color=backcolor, fontsize=25, \
            horizontalalignment='center', verticalalignment='center')
    ax.fill([0, -max_dist, -max_dist, 0], \
            [0, 0, max_dist, max_dist], \
            color=quad_codes['Consumption disamenity'], \
            ec='none', alpha=bg_alpha)
    # Lower left
    plt.text(-max_dist * 0.5, -max_dist * 0.75,  \
            'Production \ndisamenity', \
            color=backcolor, fontsize=25, \
            horizontalalignment='center', verticalalignment='center')
    ax.fill([0, -max_dist, -max_dist, 0], \
            [0, 0, -max_dist, -max_dist], \
            color=quad_codes['Production disamenity'], \
            ec='none', alpha=bg_alpha)
    # Lower right
    plt.text(max_dist * 0.5, -max_dist * 0.75,  \
            'Consumption \namenity', \
            color=backcolor, fontsize=25, \
            horizontalalignment='center', verticalalignment='center')
    ax.fill([0, max_dist, max_dist, 0], \
            [0, 0, -max_dist, -max_dist], \
            color=quad_codes['Consumption amenity'], \
            ec='none', alpha=bg_alpha)
    return ax

def directions_bu(coefs, src=2000, tgt=2010, var='manufacturing', ax=None,
        indexer='metarea', monochrome=False):
    '''
    Plot with transitions zeroed-in to origin
    '''
    cm = mpl.cm.Set1
    f = None
    if not ax:
        f, ax = plt.subplots(1)
    t = coefs[coefs['year'] == src].set_index(indexer)
    tp1 = coefs[coefs['year'] == tgt].set_index(indexer)
    vecs = t.join(tp1, lsuffix='_t', rsuffix='_tp1')\
        [['we'+var+'_t', 're'+var+'_t', 'we'+var+'_tp1', 're'+var+'_tp1']]\
        .dropna()
    colors = None
    alphas = None
    if vecs.shape[0] > 0:
        vecsZ = vecs[['we'+var+'_tp1']].sub(vecs['we'+var+'_t'], axis=0)
        vecsZ['re'+var+'_tp1'] = vecs['re'+var+'_tp1'] - vecs['re'+var+'_t']
        vecsZ['we'+var+'_t'] = 0
        vecsZ['re'+var+'_t'] = 0
        vecsZL = vecsZ.apply(_liner, axis=1, var=var).tolist()
        x1 = vecsZ['re'+var+'_tp1']
        y1 = vecsZ['we'+var+'_tp1']
        max_dist = np.sqrt((x1)**2 + (y1)**2).max()#Dists from origin (0, 0)
        xy1 = np.hstack((x1[:, None], y1[:, None]))
        lengths = _length(xy1)
        alphas = lengths * 1. / lengths.max()
        angles = np.apply_along_axis(_angler, 1, xy1)
        c = map(cm, angles / 360.)
        colors = pd.Series(c, index=vecsZ.index)
        if monochrome:
            lines = LineCollection(vecsZL, color='k', alpha=0.5)
        else:
            lines = LineCollection(vecsZL, color=c)
        ax.add_collection(lines)
        ax.set_xlim((-max_dist, max_dist))
        ax.set_ylim((-max_dist, max_dist))
        ax.set_xticklabels('')
        ax.set_xticks([])
        ax.set_yticklabels('')
        ax.set_yticks([])
        backcolor = '0.6'
        ax.set_xlabel("<-- Disamenity | Amenity -->", size=25, color=backcolor)
        ax.set_ylabel("<-- Cons. | Prod. -->", size=25, color=backcolor)
        ax.set_title("%i - %i"%(src, tgt), color=backcolor, size=30)
        ax.set_axis_bgcolor(direc_bg)
        plt.setp(ax.spines.values(), color=backcolor)
    if f:
        plt.show()
    return ax, colors, alphas

def level_maps(coefs, norm=False, duct='alpha', folder_out=None):
    '''
    Generate a level map for every variable in every year with data
    '''
    for var in vars:
        vdat = coefs.set_index('metarea')[['year', 're'+var, 'we'+var]]
        years = np.sort(vdat['year'].unique())
        maxLen = None
        if norm:
            # Calculate lengths for all years and get
            var_lengths = _length(vdat[['re'+var, 'we'+var]].dropna().values)
            maxLen = var_lengths.max()
        for y in years:
            vydat = vdat[vdat['year']==y].drop('year', axis=1)
            saveto = None
            if folder_out:
                saveto = folder_out + str(var) + '_' + str(y) + '.png'
            if len(vydat) > 0:
                _ = level_map(coefs, year=y, var=var, maxLen=maxLen, \
                        saveto=saveto, duct=duct)
    return None

def level_map(coefs, year=2010, var='manufacturing', ax=None, maxLen=None,
        saveto=None, duct='alpha'):
    '''
    Obtain a choropleth basing the color of a city in the angle of the wage
    and rent *levels* in a given year

    duct    : 'alpha'/'size'/'int'
              list with dimensions to modify based on the strength of the effect

                * 'alpha': transparency
                * 'size': dot size
                * 'int': intensity of the color
    '''
    cm = mpl.cm.Set1
    db = coefs[coefs['year']==year]\
            .set_index('metarea')\
            [['re'+var, 'we'+var]]\
            .dropna()
    if len(db) > 0:
        lengths = _length(db.values)
        if not maxLen:
            maxLen = lengths.max()
        angles = np.apply_along_axis(_angler, 1, db.values)
        c = map(cm, angles / 360.)
        colors = pd.Series(c, index=db.index)
        q_colors = pd.Series(angles, index=db.index).apply(_quad_coder)
        weights = lengths * 1. / maxLen
        if 'alpha' in duct:
            alphas = weights
        else:
            alphas = np.ones(colors.shape[0])
        if 'size' in duct:
            dotsize = weights
            dotsize = rescaler(dotsize, 10, 80)
        else:
            dotsize = 60
        if 'int' in duct:
            q_colors = pd.Series([gradienter(c, w) for c, w in zip(q_colors, weights)], \
                    index=q_colors.index)
        else:
            pass
        f = None
        if not ax:
            f, ax = plt.subplots(1)
        _ = choro(q_colors, alphas, ax, s=dotsize)
        ax.set_title("%i"%(year), color='0.6', size=30)
        if saveto:
            plt.savefig(saveto)
        if f:
            plt.show()
        return ax
    else:
        print "Empty frame"

def _quad_coder(deg):
    if 0 <= deg < 90:          # prod am
        return quad_codes['Production amenity']

    elif 90 <= deg < 180:      # cons am
        return quad_codes['Consumption amenity']

    elif 180 <= deg < 270:     # prod dis
        return quad_codes['Production disamenity']

    elif 270 <= deg < 360:     # cons dis
        return quad_codes['Consumption disamenity']
    else:
        return None

def rescaler(x, a=20, b=60):
    '''
    Re-scale a NxNone array so it ranges between [a, b]
    From: http://stackoverflow.com/a/5295202/1743507
    '''
    min = x.min()
    max = x.max()
    num = (b - a) * (x - min)
    den = max - min
    return a + (num / den)

def gradienter(c, w, floor=(1., 1., 1., 1.,)):
    '''
    Take a full color `c`, create a gradient scale between `floor` and `c`,
    and return the color in the gradient corresponding to `w`
    '''
    grad = mpl.colors.LinearSegmentedColormap.from_list('gradient', \
            [floor, c])
    return grad(w)

quad_codes = {
        'Consumption amenity': (107./255, 142./255, 35./255, 1), #green
        #'Consumption amenity': (0., 1., 0., 1), #green
        'Production amenity': (100./255, 149./255, 237./255, 1), #blue
        #'Production amenity': (0., 0., 1., 1), #blue
        'Production disamenity': (139./255, 69./255, 19./255, 1), #brown
        #'Production disamenity': (0., 0., 0., 1), #black
        'Consumption disamenity': (147./255, 112./255, 219./255, 1), #Purple
        #'Consumption disamenity': (1., 0., 0., 1), #red
        }
white = (1., 1., 1., 1.)

#blue for production amenities, green for consumption amenities, red for production disamenities

def choro(colors, alphas=None, ax=None, s=60):
    '''
    Choropleth of MSAs in `colors.index` with `colors` as color
    '''
    if not ax:
        f, ax = plt.subplots(1, facecolor='w')
    c = np.array(colors.dropna().values.tolist())
    c = pd.DataFrame(c, index=colors.dropna().index)
    if type(alphas) == np.ndarray:
        c.ix[:, 3] = alphas
    _ = _add_boundary(ax, facec=direc_bg, edgec='0.6', alpha=1)
    ipums = _add_ipums(ax, c, s=s)

    return ax

def _angle(u, v):
    '''
    Angle bewtween vectors `u` and `v` in Euclidean space (both as pairs of
    coordinates)
    NOTE: it includes a hack to get 360deg. instead of 180 only that relies on
          default in `_angler`
        |
     4  |   1
     --------
     3  |   2
        |
    ...

    Refs:
        - http://en.wikipedia.org/wiki/Dot_product
        - http://www.mathworks.co.uk/help/symbolic/mupad_ref/linalg-angle.html
    '''
    dp = np.dot(u, v)
    mu = np.sqrt(np.dot(u, u))
    mv = np.sqrt(np.dot(v, v))
    angle = degrees(acos(dp * 1. / (mu * mv)))
    if v[0] < 0: # If on the left side
        angle = 360 - angle
    return angle

_angler = lambda v: _angle(np.array([0, 1]), v)

def _length(xys):
    '''
    Return the length from origin to `xy` by calculating the hypotenuse
    ...

    Arguments
    ---------
    xys     : ndarray
              Nx2 matrix with XY coordinates of points

    Refs:

        - https://en.wikipedia.org/wiki/Hypotenuse
    '''
    x = xys[:, 0]
    y = xys[:, 1]
    hyp = np.sqrt(x**2 + y**2)
    return hyp

def movements_grid(coefs, vars=vars):
    f, axs = plt.subplots(2, len(vars), figsize=(24, 9))
    for i in range(len(vars)):
        ax = axs[0, i]
        movements(coefs, vars[i], ax)
        ax.set_title(vars[i], size=30)
        axSQ = axs[1, i]
        movements(coefs, 'sq'+vars[i], axSQ)
    plt.show()
    return vars

def movements(coefs, var='valmaxtemp', ax=None):
    years = coefs['year'].unique()
    years.sort()

    cm = mpl.cm.Spectral
    minX, maxX, minY, maxY = [], [], [], []
    f = None
    if not ax:
        f, ax = plt.subplots(1, figsize=(9, 6))
    for i in range(len(years)-1):
        t = coefs[coefs['year'] == years[i]].set_index('metarea')
        tp1 = coefs[coefs['year'] == years[i+1]].set_index('metarea')
        vecs = t.join(tp1, lsuffix='_t', rsuffix='_tp1')\
            [['we'+var+'_t', 're'+var+'_t', 'we'+var+'_tp1', 're'+var+'_tp1']]\
            .dropna()
        if vecs.shape[0] > 0:
            vecsLS = vecs.apply(_liner, axis=1, var=var).tolist()
            c = cm(np.linspace(0, 1, len(years)-1)[i])
            vecsL = LineCollection(vecsLS, color=c, label='%s-%s'%(years[i], years[i+1]))
            ax.scatter(vecs['re'+var+'_t'], vecs['we'+var+'_t'], s=0)
            ax.scatter(vecs['re'+var+'_tp1'], vecs['we'+var+'_tp1'], s=0)
            ax.add_collection(vecsL)
    ax.set_xticklabels('')
    ax.set_xticks([])
    ax.set_yticklabels('')
    ax.set_yticks([])
    ax.set_xlabel("<-- Disamenity | Amenity -->")
    ax.set_ylabel("<-- Consumption | Production -->")
    ax.legend(frameon=False, loc=0, prop={'size':6})
    ax.set_title(var)
    if f:
        plt.show()
    return None

def _liner(row, var):
    orig = np.array([row['re'+var+'_t'], row['we'+var+'_t']])
    dest = np.array([row['re'+var+'_tp1'], row['we'+var+'_tp1']])
    return [orig, dest]

def scatters_grid(coefs, vars=vars):
    f, axs = plt.subplots(2, len(vars), figsize=(21, 6))
    for i in range(len(vars)):
        ax = axs[0, i]
        scatter_by_year(coefs, vars[i], ax)
        ax.set_title(vars[i], size=30)
        axSQ = axs[1, i]
        scatter_by_year(coefs, 'sq'+vars[i], axSQ)
    plt.show()
    return vars

def scatter_by_year(coefs, var='valmaxtemp', ax=None):
    if not ax:
        f, ax = plt.subplots(1, figsize=(8, 6))
    cm = mpl.cm.Spectral
    g = coefs.groupby('year')
    for i, (y, gr) in enumerate(g):
        gr = gr[['we'+var, 're'+var]]
        if gr.dropna().shape[0] > 0:
            c = i * 1. / coefs.year.unique().shape[0]
            sc = ax.scatter(gr['re'+var], gr['we'+var], c=cm(c),
                    edgecolor='none', label=y)
        else:
            #print "Skipping %s in %s"%(var, y)
            pass
    ax.axvline(0, c='0.5')
    ax.axhline(0, c='0.5')
    ax.set_xlabel('re'+var)
    ax.set_ylabel('we'+var)
    ax.legend(frameon=False, loc=0, prop={'size':6})
    plt.tight_layout()
    if not ax:
        plt.show()
    return ax

def cs_grid(coefs, grouper='year', transVar=None, norm=False, saveto=None):
    '''
    Grid of cross-sections
    ...

    Arguments
    ---------
    coefs
    grouper
    transVar
    norm
    saveto

    '''
    def _add_legend():
        # Legend (works but involves some hacking)
        lg = coefs[['color', 'cat']].drop_duplicates().dropna()
        lg['x'] = 0
        lg['y'] = 0
        ax0 = f.add_subplot(s, s, 0)
        scs = []
        for id, row in lg.iterrows():
            sc = ax0.scatter(row['x'], row['y'], c=row['color'], label=row['cat'],
                    visible=True, edgecolor='none', alpha=0.75, s=50)
            scs.append(sc)
        ax0.axes.get_yaxis().set_visible(False)
        ax0.axes.get_xaxis().set_visible(False)
        ax0.set_frame_on(False)
        ax0.legend(mode='expand', scatterpoints=1, frameon=False)
        for sc in scs:
            sc.set_visible(False)
        return None

    s = int(np.round(np.sqrt(coefs['year'].unique().shape[0])))

    f = plt.figure(figsize=(12, 9))
    f.set_facecolor('w')

    maxalpha = None
    if transVar and norm:
        maxalpha = coefs[transVar].max()

    for i, (year, ydb) in enumerate(coefs.groupby(grouper)):
        ax = f.add_subplot(s, s, i+1)
        ax = cross_section(ydb, ax, title='Year %i'%year, \
                transVar=transVar, maxalpha=maxalpha)

    _ = _add_legend()

    if saveto:
        if transVar:
            saveto = saveto.replace('.png', '_%s.png'%transVar)
        if norm:
            saveto = saveto.replace('.png', '_norm.png')
        plt.savefig(saveto)
        plt.show()
        plt.close('all')

    return None

def cross_section(coefs, ax=None, title=None, transVar=None, maxalpha=None, square=True):
    '''
    Generate cross-sectional plot
    ...

    Arguments
    ---------
    coefs
    ax
    title
    transVar
    minmax
    square

    '''
    if not ax:
        f, ax = plt.subplots(1, facecolor='w')

    _ = _add_boundary(ax)

    ydb = coefs.dropna()
    if ydb.shape[0] > 0:
        for color, gr in ydb.groupby('color'):
            if transVar:
                r, g, b, a = cc.to_rgba(color)
                trans = np.abs(gr[transVar])
                if not maxalpha:
                    maxalpha = trans.max()
                if square:
                    trans = trans**2
                    maxalpha = maxalpha**2
                color = trans.apply(lambda x: (r, g, b, x*1./maxalpha))\
                        .tolist()
            ax.scatter(gr['x'], gr['y'], edgecolor='none', c=color)
    ax.set_title(title, color='0.6')
    return ax

def _add_boundary(ax, facec='0.6', edgec='0.3', alpha=0.25):
    states_link = 'data/states2163_light.shp'
    states_shp = ps.open(states_link)
    patchco = viz.map_poly_shp(states_shp)
    patchco.set_facecolor(facec)
    patchco.set_edgecolor(edgec)
    patchco.set_linewidth('0.25')
    patchco.set_alpha(alpha)
    ax = viz.setup_ax([patchco], ax)

def _add_ipums(ax, colors=None, s=60, ipums_link='data/ipums_metarea.csv'):
    '''
    `colors` need to be a pd.Series indexed on `metarea`
    '''
    ipums = pd.read_csv(ipums_link).set_index('metarea')
    if type(colors) == pd.DataFrame:
        ipums_algd = ipums.reindex(colors.index)
        sc = ax.scatter(ipums_algd['x'], ipums_algd['y'], edgecolor='none', \
                c=colors.values, s=s)
    else:
        sc = ax.scatter(ipums['x'], ipums['y'], edgecolor='none', alpha=0.75, s=s)
    return sc

def reproject_ipums():
    '''
    Re-project ipums shapefile and write out to csv for quicker plotting
    ...

    Arguments
    ---------
    db          : DataFrame
                  DataFrame with coordinates to be reprojected
    lat         : str
                  Column name in db for lattitude
    lon         : str
                  Column name in db for longitude
    Returns
    -------
    db          : DataFrame
                  Original DataFrame to which columns 'x' and 'y' have been
                  added with projected coordinates
    '''
    ipums_list = 'data/ipums_wgs84.shp'
    xys = np.array([pt.centroid for pt in ps.open(ipums_list)])
    ids = dbf2df(ipums_list.replace('.shp', '.dbf'))['id'].values
    #-------------------------------
    from osgeo import osr
    orig = osr.SpatialReference()
    orig.SetWellKnownGeogCS("WGS84")
    target = osr.SpatialReference()
    target.ImportFromEPSG(2163) #http://spatialreference.org/ref/epsg/2163/
    trCRS = osr.CoordinateTransformation(orig, target)

    prjd_xys = np.array(trCRS.TransformPoints(xys.tolist()))[:, :2]
    db = pd.DataFrame({'lon': xys[:, 0], 'lat': xys[:, 1], \
            'x': prjd_xys[:, 0], 'y': prjd_xys[:, 1]}, index=ids)
    db.index.name = 'pid'
    db.to_csv('data/ipums.csv')
    return db

def load_coefs(link, var='valmaxtemp'):
    '''
    Load coefficients from Michiel's regressions and match them up to
    coordinates
    '''
    coefs = pd.read_csv(link)
    coefs['cat'] = coefs.apply(_roback_decider, axis=1, var=var)
    coefs['color'] = coefs['cat'].apply(_color_picker)
    '''
    coefs['metid'] = coefs['statefip'].apply(lambda x: str(x).zfill(2)) + \
            '-' + coefs['metarea'].apply(lambda x: str(x).zfill(4))
    '''

    # Link own codes pSS-PPPP to official metarea codes pSS-MMMM
    # Crosswalk was provided by Michiel in a Nov. 2012 email
    walk = pd.read_csv('data/pumametarea.csv')
    walk['metid'] = walk['statefip'].apply(lambda x: str(x).zfill(2)) + \
            '-' + walk['metarea'].apply(lambda x: str(x).zfill(4))
    walk['pid'] = 'p' + walk['statefip'].apply(lambda x: str(x).zfill(2)) + \
            '-' + walk['puma'].apply(lambda x: str(x).zfill(5))

    ipums = pd.read_csv('data/ipums.csv', index_col=0)
    ipums = ipums.join(walk.set_index('pid', drop=False)[['metarea', 'pid']])

    coefs = coefs.join(ipums.set_index('metarea'), on='metarea')
    # Keep only observations for which we have a location
    coefs = coefs.reindex(coefs['pid'].dropna().index)
    return coefs

def _roback_decider(row, var='valmaxtemp'):
    effect = side = None
    if (row['we'+var] > 0) and (row['re'+var] > 0):
        return 'production-amenity'
    if (row['we'+var] > 0) and (row['re'+var] < 0):
        return 'consumption-disamenity'
    if (row['we'+var] < 0) and (row['re'+var] > 0):
        return 'consumption-amenity'
    if (row['we'+var] < 0) and (row['re'+var] < 0):
        return 'production-disamenity'

def _color_picker(cat):
    if cat == 'production-amenity':
        return 'red'
    elif cat == 'consumption-amenity':
        return 'blue'
    elif cat == 'consumption-disamenity':
        return 'purple'
    elif cat == 'production-disamenity':
        return 'brown'
    else:
        return None

def quadrants(db, year, saveto=None):
    tmp = db[db['year']==year].dropna()
    f, ax = plt.subplots(1)
    sc = ax.scatter(tmp['clwage'], tmp['clrent'], linewidth=0)
    plt.axvline(0, c='k', alpha=0.5)
    plt.axhline(0, c='k', alpha=0.5)
    ax.set_xlabel('clwage')
    ax.set_ylabel('clrent')
    plt.title('Year %i | %i cities'%(year, tmp.shape[0]))
    if saveto:
        plt.savefig(saveto)
    else:
        plt.show()

def build_storyboard(var, folder):
    '''
    Build a `matplotlib` figure with all the images of a storyboard in the
    right order
    '''
    years = [1970, 1980, 1990, 2000, 2010]
    f, axs = plt.subplots(3, 3, figsize=(12, 12))
    axs = axs.flatten()
    for i, y in zip(range(0, len(axs), 2), range(len(years)-1)):
        sn = folder + var + '_' + str(years[y]) + '.png'
        axs[i].imshow(plt.imread(sn))
        axs[i].axes.get_yaxis().set_visible(False)
        axs[i].axes.get_xaxis().set_visible(False)
        axs[i].set_frame_on(False)
        tr = folder + var + '_' + str(years[y]) + '-' + str(years[y+1]) + '.png'
        axs[i+1].imshow(plt.imread(tr))
        axs[i+1].axes.get_yaxis().set_visible(False)
        axs[i+1].axes.get_xaxis().set_visible(False)
        axs[i+1].set_frame_on(False)
    y = folder + var + '_' + '2010.png'
    axs[-1].imshow(plt.imread(y))
    axs[-1].axes.get_yaxis().set_visible(False)
    axs[-1].axes.get_xaxis().set_visible(False)
    axs[-1].set_frame_on(False)
    plt.tight_layout()
    plt.show()
    return None
