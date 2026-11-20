"""Regenerate the figures of the documentation.

    python docs/make_figures.py quickstart DMU_PICKLE
    python docs/make_figures.py windows INPUT_POP_TRAJ

``DMU_PICKLE`` is the ``*_mbar_dmu_estimates.pkl`` file of the quickstart
run and ``INPUT_POP_TRAJ`` the population written by ``catemc generate`` for
``examples/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/111/config.yaml``. Each figure is
written in a light and a dark version to docs/source/_static/figures.
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'source', '_static', 'figures')

THEMES = {
    'light': dict(surface='#ffffff', ink='#0b0b0b', secondary='#52514e',
                  muted='#898781', grid='#e1e0d9', axis='#c3c2b7',
                  series=['#2a78d6', '#eb6834', '#1baf7a', '#eda100'],
                  ramp=['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b']),
    'dark': dict(surface='#131416', ink='#f0efec', secondary='#c3c2b7',
                 muted='#898781', grid='#2c2c2a', axis='#383835',
                 series=['#3987e5', '#d95926', '#199e70', '#c98500'],
                 ramp=['#0d366b', '#1c5cab', '#3987e5', '#86b6ef', '#cde2fb']),
}


def style(ax, theme):
    ax.set_facecolor(theme['surface'])
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(theme['axis'])
    ax.tick_params(colors=theme['muted'], labelsize=9, length=3)
    ax.xaxis.label.set_color(theme['secondary'])
    ax.yaxis.label.set_color(theme['secondary'])


def quickstart(path):
    with open(path, 'rb') as handle:
        dmu = pickle.load(handle)
    temperatures = sorted(dmu)
    metals = ['Ni', 'Pd', 'Pt', 'Au']
    for mode, theme in THEMES.items():
        fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=200)
        fig.patch.set_facecolor(theme['surface'])
        style(ax, theme)
        ax.axhline(0, color=theme['axis'], lw=1)
        ax.grid(axis='y', color=theme['grid'], lw=0.6)
        ax.set_axisbelow(True)
        ends = {}
        for metal, color in zip(metals, theme['series']):
            values = [float(dmu[T][f'Cu2{metal}']) for T in temperatures]
            ax.plot(temperatures, values, color=color, lw=2, marker='o',
                    ms=5, mec=theme['surface'], mew=1.2, label=metal,
                    clip_on=False)
            ends[metal] = values[-1]
        # direct labels at the line ends, pushed apart when they collide
        order = sorted(ends, key=ends.get)
        y_label, gap = {}, 0.028
        for i, metal in enumerate(order):
            y = ends[metal]
            if i and y - y_label[order[i - 1]] < gap:
                y = y_label[order[i - 1]] + gap
            y_label[metal] = y
        for metal in metals:
            ax.annotate(f'{metal}', xy=(temperatures[-1], ends[metal]),
                        xytext=(temperatures[-1] + 12, y_label[metal]),
                        color=theme['ink'], fontsize=9.5, va='center')
        ax.annotate('Cu (reference)', xy=(temperatures[0], 0),
                    xytext=(temperatures[0], 0.012), color=theme['muted'],
                    fontsize=9, va='bottom')
        ax.set_xlim(temperatures[0] - 10, temperatures[-1] + 10)
        ax.set_xticks(temperatures)
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel(r'$\mu_m - \mu_{\mathrm{Cu}}$ (eV)')
        legend = ax.legend(ncol=4, frameon=False, fontsize=9,
                           loc='lower left', bbox_to_anchor=(0, 1.0),
                           handlelength=1.6, columnspacing=1.4)
        for text in legend.get_texts():
            text.set_color(theme['secondary'])
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'quickstart_dmu_{mode}.png'),
                    facecolor=theme['surface'])
        plt.close(fig)


def windows(path):
    from ase.io import read
    pop = read(path, index=':')
    points, window_of, n_ads, starts = [], [], [], []
    for i, atoms in enumerate(pop):
        count = sum(1 for a in atoms if a.symbol in ('N', 'H'))
        starts.append(atoms.info['data']['mu'])
        for mu in atoms.info['data']['mu_window']:
            points.append(mu)
            window_of.append(i)
            n_ads.append(count)
    points = np.round(np.array(points, dtype=float), 6)
    xs, ys = np.unique(points[:, 0]), np.unique(points[:, 1])
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    ix = np.searchsorted(xs, points[:, 0])
    iy = np.searchsorted(ys, points[:, 1])
    win = -np.ones((len(ys), len(xs)), dtype=int)
    val = np.zeros((len(ys), len(xs)))
    win[iy, ix] = window_of
    val[iy, ix] = n_ads
    for mode, theme in THEMES.items():
        fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=200)
        fig.patch.set_facecolor(theme['surface'])
        style(ax, theme)
        for side in ('left', 'bottom'):
            ax.spines[side].set_visible(False)
        cmap = LinearSegmentedColormap.from_list('ramp', theme['ramp'])
        mesh = ax.pcolormesh(xs, ys, val, cmap=cmap, shading='nearest',
                             vmin=0, vmax=val.max())
        # a surface-coloured gap between neighbouring windows
        for j in range(len(ys)):
            for i in range(len(xs)):
                if i + 1 < len(xs) and win[j, i] != win[j, i + 1]:
                    x = xs[i] + dx / 2
                    ax.plot([x, x], [ys[j] - dy / 2, ys[j] + dy / 2],
                            color=theme['surface'], lw=2.2,
                            solid_capstyle='projecting')
                if j + 1 < len(ys) and win[j, i] != win[j + 1, i]:
                    y = ys[j] + dy / 2
                    ax.plot([xs[i] - dx / 2, xs[i] + dx / 2], [y, y],
                            color=theme['surface'], lw=2.2,
                            solid_capstyle='projecting')
        starts = np.array(starts, dtype=float)
        ax.scatter(starts[:, 0], starts[:, 1], s=13, color=theme['ink'],
                   edgecolors=theme['surface'], linewidths=0.8, zorder=3,
                   label='starting point of a replica')
        ax.set_xlabel(r'$\Delta\mu_{\mathrm{N}}$ (eV)')
        ax.set_ylabel(r'$\Delta\mu_{\mathrm{H}}$ (eV)')
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(-1, 0.01, 0.25))
        ax.set_yticks(np.arange(-1, 0.01, 0.25))
        legend = ax.legend(frameon=False, fontsize=9, loc='lower left',
                           bbox_to_anchor=(-0.02, 1.0), handletextpad=0.3)
        for text in legend.get_texts():
            text.set_color(theme['secondary'])
        bar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        bar.outline.set_visible(False)
        bar.ax.tick_params(colors=theme['muted'], labelsize=9, length=3)
        bar.set_label('N and H atoms in the starting structure',
                      color=theme['secondary'], fontsize=9.5)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'mu_windows_{mode}.png'),
                    facecolor=theme['surface'])
        plt.close(fig)


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in ('quickstart', 'windows'):
        sys.exit(__doc__)
    os.makedirs(OUT, exist_ok=True)
    {'quickstart': quickstart, 'windows': windows}[sys.argv[1]](sys.argv[2])
