import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks
import seaborn as sns
import re

plt.rc('font', family='serif')
plt.rc('text', usetex=True)

def dumpfile_to_df(dumpfile):
    '''
    Reads lammps dump file to data frame
    '''
    with open(dumpfile) as f:
        text = f.read()
    m = re.search('ITEM: BOX BOUNDS pp pp pp\n0.0000000000000000e\+00\s+(\S+)\n0.0000000000000000e\+00\s+(\S+)\s+0.0000000000000000e\+00\s+(\S+)', text)
    latvec = [
        [float(m.group(1)), 0.0, 0.0],
        [0.0, float(m.group(2)), 0.0],
        [0.0, 0.0, float(m.group(3))]
        ]

    lines = [l.split() for l in text.split('\n')[9:] if l]
    df = pd.DataFrame(lines).astype(np.float64)
    df.columns = ['atom_id', 'atom_type', 'x', 'y', 'z', 'c_csym', 'stress_1', 'stress_2', 'stress_3', 'stress_4', 'stress_5', 'stress_6', 'energy']
    df['atom_id'] = df['atom_id'].astype(int)
    df['atom_type'] = df['atom_type'].astype(int)
    df = df.sort_values(by='atom_id').reset_index(0, drop=True)

    return df, latvec

def scatterplotter(x, y, z, title, colorbar_label, output, strip_width=None, vmin=None, vmax=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 5))
    p = plt.scatter(x, y, c=z, cmap='jet', s=2, alpha=0.8, lw=0)

    cax = fig.add_axes([
        ax.get_position().x1+0.01 + 0.1,
        ax.get_position().y0 + 0.05,
        0.02,
        ax.get_position().height
        ])
    cbar = plt.colorbar(p, cax=cax)
    cbar.set_label(colorbar_label, rotation=90, fontsize=9, fontweight='bold', labelpad=8)
    cbar.ax.tick_params(labelsize=9)
    ax.text(0.1, 0.9, title, transform=ax.transAxes)
    if strip_width is not None:
        ax.axvspan(0, strip_width, alpha=0.5, color='#2f2f2f')
    vmin = vmin if vmin is not None else np.min(z)
    vmax = vmax if vmax is not None else np.max(z)
    plt.clim(vmin=vmin, vmax=vmax)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    ax.set_aspect('equal', 'box')
    ax.set(xlabel='$x$ (ang)', ylabel='$y$ (ang)')
    fig.tight_layout()
    plt.savefig(output, bbox_inches='tight')
    plt.close()

# def lineplotter(x, y_ouyang, y_refit, xlabel, ylabel, output, plot=True):
#     x_ouyang1, x_ouyang2 = find_sp_peaks(x, y_ouyang)
#     x_refit1, x_refit2 = find_sp_peaks(x, y_refit)

#     width_ouyang = x_ouyang2 - x_ouyang1
#     width_refit = x_refit2 - x_refit1

#     if plot:
#         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
#         colors = sns.color_palette()

#         ax.plot(x, y_refit, '-', color=colors[0], label=f'Refit: $W_\\mathrm{{D}}$ = {width_refit:.1f} ang')
#         for peak_x in [x_refit1, x_refit2]:
#             ax.axvline(x=peak_x, c=colors[0], ls='--')

#         ax.plot(x, y_ouyang, '-', color=colors[1], label=f'Ouyang: $W_\\mathrm{{D}}$ = {width_ouyang:.1f} ang')
#         for peak_x in [x_ouyang1, x_ouyang2]:
#             ax.axvline(x=peak_x, c=colors[1], ls='--')

#         ax.set(ylim=(None, 0.25))
#         ax.legend(loc='upper right', fancybox=False, edgecolor='k')
#         ax.set(xlabel=xlabel, ylabel=ylabel)
#         fig.tight_layout()
#         plt.savefig(output, bbox_inches='tight')
#         plt.close()

#     return width_ouyang, width_refit

def rotate_coords(coords, theta):
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
        ])
    rotated = rotation.dot(coords.T).T
    return rotated

def plot_energy(twist_angle, potential, label, atom_type=1, strip_width=None):
    d_all, latvec = dumpfile_to_df(f"kc_rebo_{potential}/raw/simulations/{twist_angle}/dump_final.txt")

    # select only one layer, e.g. for the bottom layer, `atom_type` is 1
    d = d_all.loc[d_all.atom_type == atom_type, :].reset_index(0, drop=True)

    # divide the angle by two because the geometry is setup such that
    # the top layer is twisted counter-clockwise by `twist_angle`/2,
    # and the bottom layer is twisted clockwise by `twist_angle`/2
    theta = float(twist_angle.replace('-', '.'))/2 *np.pi/180
    d_rotated = d.copy()
    d_rotated[['x', 'y', 'z']] = rotate_coords(d[['x', 'y', 'z']], theta)
    d_rotated['potential'] = potential

    scatterplotter(d_rotated.x, d_rotated.y, d_rotated.energy, label, 'Energy (eV/atom)', f'{twist_angle}_{potential}_energy.pdf',
        strip_width=strip_width, vmin=-7.427, vmax=-7.412)
    return d_rotated

def plot_energy_strip(d, output):
    g = sns.FacetGrid(data=d, hue='potential')
    colors = sns.color_palette()
    for i, potential in enumerate(g.hue_names):
        dd = d.loc[d.potential == potential, :]
        g.ax.plot(dd.y, dd.energy, 'o-', label=potential, color=colors[i], ms=3, lw=1, mec='white', mew=0.1)
    g.ax.legend(bbox_to_anchor=(0.45, 0.72), frameon=False, fontsize=8)
    g.set(
        xlabel='Distance along the $y$-axis ($\\mathrm{\\AA}$)',
        ylabel='Energy (eV/atom)'
        )
    g.fig.set_size_inches(3, 3)
    plt.savefig(output, bbox_inches='tight')

if __name__ == '__main__':
    widths = []
    for twist_angle in ['0-99']:
        strip_width = 0.25
        d_ouyang = plot_energy(twist_angle, 'ouyang', '(a) Ouyang', strip_width=strip_width)
        d_refit = plot_energy(twist_angle, 'refit', '(b) Refit', strip_width=strip_width)
        d = pd.concat([d_ouyang, d_refit], ignore_index=True)
        d_strip = d.loc[(-strip_width <= d.x) & (d.x < strip_width), :]
        print(d_strip.shape)
        plot_energy_strip(d_strip, f'{twist_angle}_energy_strip.pdf')
