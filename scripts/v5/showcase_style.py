#!/usr/bin/env python3
"""One look for every showcase figure: white paper, one palette, one set of board primitives.

Kept apart from the figures so a colour or a line weight is changed in ONE place, and so the board
(pegs, keep-out rings, goal line, hose) is drawn identically wherever it appears -- a figure, a video
frame, or an inset.
"""
from __future__ import annotations

import numpy as np

PAPER = '#FFFFFF'
INK = '#16181D'
MUTED = '#6B7280'
FAINT = '#9AA1AC'
GRID = '#E8EAED'
PANEL = '#F7F8FA'

HOSE = '#C2352E'          # the tracked hose, after the move
BEFORE = '#B9BEC6'        # the tracked hose, before the move
GOAL = '#2E6FBF'          # the goal centreline
MOVE = '#1F9D63'          # a commanded gripper path
PEG = '#E3A12C'           # a peg and its keep-out ring
PRED = '#7C3AED'          # what the learned model forecast
GOOD = '#0F766E'          # the best end of the candidate scale
BAD = '#C9D1D9'           # the worst end of the candidate scale
FLAG = '#B91C1C'
GOAL_DARK = '#7EC8F5'     # the goal line drawn ON the photograph: the plot blue vanishes on a dark table

CANDIDATE_COLORS = ['#0B3B57', '#0F766E', '#5EAAA8', '#A9C0C4', '#D3DADF']

# Proposal families, used only where the semantic colours above do not appear (the statistics panels).
FAMILY_COLORS = {'goal_carry': '#0F766E', 'crossing_carry': '#0369A1', 'bimanual_carry': '#B45309',
                 'single_carry': '#DB2777', 'tail_sweep': '#0EA5E9', 'release': '#9AA6B2'}
PEG_TINTS = ['#7A4E05', '#C08A1E', '#E0B667']


def rc():
    """The rcParams every showcase figure sets, once."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.facecolor': PAPER, 'savefig.facecolor': PAPER, 'axes.facecolor': PAPER,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Nimbus Sans', 'Liberation Sans', 'Noto Sans', 'DejaVu Sans'],
        'font.size': 9.5, 'axes.titlesize': 10.5, 'axes.labelsize': 9.5,
        'axes.titleweight': 'medium', 'axes.labelcolor': INK, 'axes.edgecolor': '#C7CCD3',
        'axes.linewidth': .8, 'axes.titlepad': 7, 'axes.labelpad': 4,
        'axes.spines.top': False, 'axes.spines.right': False,
        'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.major.width': .8, 'ytick.major.width': .8,
        'grid.color': GRID, 'grid.linewidth': .7, 'legend.frameon': False, 'legend.fontsize': 8.5,
        'legend.handlelength': 1.6, 'legend.borderaxespad': 0, 'legend.columnspacing': 1.3,
        'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
        'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': None,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    return plt


def candidate_cmap():
    """Low forecast cost = deep teal and opaque; high = pale grey. Reads on white, prints, colour-blind safe."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('candidate', CANDIDATE_COLORS)


def board(ax, data, *, xlim=None, ylim=None, pegs=True, goal=True, scale=None, frame=False, table=True):
    """The table as every panel shows it: robot at the bottom, +x up the page, +y to the LEFT.

    The rig's world frame has +x away from the robot and +y to its left, so a top-down view with the
    robot at the bottom of the page is (y, x) with y increasing leftwards -- done here once, by inverting
    the horizontal axis, so no figure has to remember it.
    """
    ax.set_aspect('equal', adjustable='box')
    lo_x, hi_x = xlim if xlim else data['extent'][1]
    lo_y, hi_y = ylim if ylim else data['extent'][0]
    if table:
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch((min(lo_x, hi_x) + .004, lo_y + .004),
                                    abs(hi_x - lo_x) - .008, hi_y - lo_y - .008,
                                    boxstyle='round,pad=0,rounding_size=.012', linewidth=0,
                                    facecolor=PANEL, zorder=.4))
    if goal:
        g = np.asarray(data['goal'], float)
        ax.plot(g[:, 1], g[:, 0], color=GOAL, lw=1.5, ls=(0, (5, 3)), zorder=2.2,
                solid_capstyle='butt')
    if pegs:
        near = float((data['metrics'].get('objective') or {}).get('near_m', .07))
        for i, p in enumerate(data['pegs']):
            ax.add_patch(_circle(ax, (p['y'], p['x']), near, facecolor='none', edgecolor=PEG,
                                 lw=.8, ls=(0, (1.6, 2.2)), alpha=.75, zorder=2.5))
            ax.add_patch(_circle(ax, (p['y'], p['x']), max(p['r'], .009), facecolor=PEG,
                                 edgecolor='#B8801F', lw=.6, zorder=3.2))
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)
    ax.invert_xaxis()
    ax.set_xticks([]), ax.set_yticks([])
    for side, spine in ax.spines.items():
        spine.set_visible(frame)
        spine.set_color('#DCE0E5')
    if scale:
        bar(ax, scale)
    return ax


def _circle(ax, centre, radius, **kw):
    from matplotlib.patches import Circle
    return Circle(centre, radius, **kw)


def extent(data, pad=.055, *, probes=False):
    """One pair of limits for EVERY panel, so shapes can be compared across figures. -> ((x0,x1), (y0,y1))

    Built from the goal, every tracked state and every state the model predicted along the way: a per-panel
    autoscale silently rescales the hose between cycles, which is exactly the comparison these figures exist
    to make.
    """
    pts = [np.asarray(data['goal'], float)]
    if probes:
        pts.append(np.asarray(data['states'], float).reshape(-1, 3))
    else:
        pts.append(np.asarray(data['final'], float))
    for c in data['cycles']:
        for key in ('pre', 'post'):
            if c.get(key) is not None:
                pts.append(np.asarray(c[key], float))
        if len(c.get('predicted', [])):
            pts.append(np.asarray(c['predicted'], float).reshape(-1, 3))
    if probes:
        for p in data['probes']:
            for key in ('pre', 'post'):
                if p.get(key) is not None:
                    pts.append(np.asarray(p[key], float))
    pts += [np.array([[p['x'], p['y'], 0.]]) for p in data['pegs']]
    v = np.concatenate(pts)
    return ((float(v[:, 0].min() - pad), float(v[:, 0].max() + pad)),
            (float(v[:, 1].min() - pad), float(v[:, 1].max() + pad)))


def limits(points, pad=.05, *, low=0., high=100.):
    """Panel limits from what a figure actually draws, optionally trimming outlier tails by percentile."""
    p = np.asarray(points, float).reshape(-1, 3)
    x = np.percentile(p[:, 0], [low, high])
    y = np.percentile(p[:, 1], [low, high])
    return ((float(x[0] - pad), float(x[1] + pad)), (float(y[0] - pad), float(y[1] + pad)))


def bar(ax, metres=.1, *, label=None, loc=(.035, .085)):
    """A scale bar in data units, placed at an axes fraction and drawn INTO the panel."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    px = x0 + (x1 - x0) * loc[0]
    py = y0 + (y1 - y0) * loc[1]
    end = px + metres * np.sign(x1 - x0)
    ax.plot([px, end], [py, py], color=MUTED, lw=1.5, solid_capstyle='butt', zorder=6)
    ax.text(.5 * (px + end), py + .008, label or '%d cm' % round(100 * metres),
            ha='center', va='bottom', fontsize=7.5, color=MUTED, zorder=6)


def hose(ax, points, *, color=HOSE, lw=3.2, alpha=1., zorder=4, label=None, radius=None, dots=False):
    """A hose centreline. With `radius` it is drawn at its true width as a soft band underneath."""
    p = np.asarray(points, float)
    if radius:
        ax.plot(p[:, 1], p[:, 0], color=color, lw=lw + 2.2 * 72 * radius * .06, alpha=.16 * alpha,
                solid_capstyle='round', zorder=zorder - .1)
    line, = ax.plot(p[:, 1], p[:, 0], color=color, lw=lw, alpha=alpha, zorder=zorder, label=label,
                    solid_capstyle='round')
    if dots:
        ax.scatter(p[:, 1], p[:, 0], s=5.5, color=color, alpha=alpha, zorder=zorder + .1,
                   linewidths=0)
    return line


def arrow(ax, path, *, color=MOVE, lw=2.0, alpha=1., zorder=5, head=.014, label=None):
    """One commanded gripper path: the waypoints joined, with a head at the end."""
    p = np.asarray(path, float)
    ax.plot(p[:, 1], p[:, 0], color=color, lw=lw, alpha=alpha, zorder=zorder, label=label,
            solid_capstyle='round')
    d = p[-1] - p[-2]
    n = np.linalg.norm(d[:2])
    if n > 1e-6:
        ax.annotate('', xy=(p[-1, 1], p[-1, 0]), xytext=(p[-1, 1] - d[1] * .28, p[-1, 0] - d[0] * .28),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, alpha=alpha,
                                    mutation_scale=9 + 320 * head, shrinkA=0, shrinkB=0),
                    zorder=zorder + .1, annotation_clip=False)
    ax.scatter([p[0, 1]], [p[0, 0]], s=26, facecolor='white', edgecolor=color, lw=1.2, alpha=alpha,
               zorder=zorder + .2)


def title(ax, text, *, sub=None, pad=5, size=10.5, weight='demibold'):
    ax.set_title(text, fontsize=size, color=INK, pad=pad + (12 if sub else 0), loc='left', weight=weight)
    if sub:
        ax.annotate(sub, xy=(0, 1), xytext=(0, pad), xycoords='axes fraction',
                    textcoords='offset points', ha='left', va='bottom', fontsize=8.3, color=MUTED)


def tidy(ax, *, y=True, x=False):
    ax.grid(axis='y' if y and not x else ('both' if y and x else 'x'), alpha=1., zorder=.5)
    ax.set_axisbelow(True)
    return ax


def label_pegs(ax, data, *, size=7.5, color='#96690F', offset=(0, 9)):
    """Peg names, stroked in white so a hose or a candidate bundle underneath cannot swallow them."""
    import matplotlib.patheffects as pe
    for i, p in enumerate(data['pegs']):
        ax.annotate('peg %d' % i, xy=(p['y'], p['x']), xytext=offset, textcoords='offset points',
                    ha='center', va='bottom', fontsize=size, color=color, zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground='white')])


def strip_figure(data, rows, *, width=10.6, left=.55, right=.28, top=.92, gap=.72, bottom=.62,
                 labels=None, extra=0.):
    """A figure of `rows` board strips, sized in INCHES from the board's own aspect.

    A top-down panel of this table is a wide, shallow strip (about 3:1). Letting matplotlib pick the
    figure size and then asking for equal aspect leaves big empty bands above and below every panel and
    squashes the titles into them, which is what the first attempt did. Here the strip height follows
    from the data, the figure follows from the strips, and every panel lands on an exact rectangle.

    labels: an optional per-row title height in inches (default `gap`). extra: inches reserved at the bottom.
    """
    plt = rc()
    (x0, x1), (y0, y1) = data['extent']
    panel_w = width - left - right
    panel_h = panel_w * (x1 - x0) / (y1 - y0)
    heads = [gap] * rows if labels is None else list(labels)
    height = top + sum(heads) + rows * panel_h + bottom + extra
    fig = plt.figure(figsize=(width, height))
    axes, y = [], height - top
    for r in range(rows):
        y -= heads[r] + panel_h
        axes.append(fig.add_axes([left / width, y / height, panel_w / width, panel_h / height]))
    return fig, axes, dict(width=width, height=height, panel_w=panel_w, panel_h=panel_h,
                           left=left, right=right, bottom=bottom, extra=extra)
