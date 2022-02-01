import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks
import seaborn as sns
import re

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 1000)
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

def rotate_coords(coords, theta):
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
        ])
    rotated = rotation.dot(coords.T).T
    return rotated

def get_data(twist_angle, potential, relaxed=True, atom_type=1):
    '''
    Creates a data frame from `twist_angle` and `potential`.
    `atom_type` (int):
        1 for the bottom layer
        2 for the top layer
    '''
    which_dump = 'final' if relaxed else 'initial'
    d_all, latvec = dumpfile_to_df(f"kc_rebo_{potential}/raw/simulations/{twist_angle}/dump_{which_dump}.txt")

    # select only one layer
    d = d_all.loc[d_all.atom_type == atom_type, :].reset_index(0, drop=True)

    # divide the angle by two because the geometry is setup such that
    # the top layer is twisted counter-clockwise by `twist_angle`/2,
    # and the bottom layer is twisted clockwise by `twist_angle`/2
    theta = float(twist_angle.replace('-', '.'))/2 *np.pi/180
    d_rotated = d.copy()
    d_rotated[['x', 'y', 'z']] = rotate_coords(d[['x', 'y', 'z']], theta)
    d_rotated['potential'] = potential
    return d_rotated

def get_disp_data(twist_angle, pot):
    '''
    Subtracts to initial coordinates from
    '''
    d_final = get_data(twist_angle, pot, relaxed=True)
    d_init = get_data(twist_angle, pot, relaxed=False)
    d_init['dx'] = d_final['x'] - d_init['x']
    d_init['dy'] = d_final['y'] - d_init['y']
    d_init['dz'] = d_final['z'] - d_init['z']
    d_init['mag'] = (d_init['dx']**2 + d_init['dy']**2)**0.5
    return d_init

def scatterplotter(fig, ax, x, y, z, title, ylabel=None, colorbar=True, strip_width=None):
    p = ax.scatter(x, y, c=z, cmap='jet', s=2, lw=0, vmin=-7.427, vmax=-7.412)
    ax.text(0.1, 1.02, title, transform=ax.transAxes)
    if strip_width is not None:
        ax.axvspan(0, strip_width, alpha=0.5, color='#2f2f2f')

    ax.set_aspect('equal', 'box')
    ax.set(
        xlabel='$x$ (ang)', xlim=(0, 140),
        ylabel=ylabel, ylim=(0, 248)
        )
    if colorbar:
        cax = fig.add_axes([
            ax.get_position().x1 + 0.11, # x-coordinate corner
            ax.get_position().y0 - 0.02, # y-coordinate corner
            0.01, # width
            ax.get_position().height*1.13 # height
            ])
        cax.set_title('Energy (eV/atom)')
        ax.figure.colorbar(p, cax=cax)

def quiverplotter(fig, ax, x, y, dx, dy, mag, title, ylabel=None, colorbar=True):
    p = ax.quiver(x, y, dx, dy, mag, cmap='jet', headlength=20, headwidth=24, headaxislength=9)
    ax.text(0.1, 1.02, title, transform=ax.transAxes)
    ax.set_aspect('equal', 'box')
    ax.set(
        xlabel='$x$ (ang)', xlim=(0, 140),
        ylabel=ylabel, ylim=(0, 248)
        )
    if colorbar:
        cax = fig.add_axes([
            ax.get_position().x1 + 0.11, # x-coordinate corner
            ax.get_position().y0 - 0.02, # y-coordinate corner
            0.01, # width
            ax.get_position().height*1.13 # height
            ])
        cax.set_title('Energy (eV/atom)')
        ax.figure.colorbar(p, cax=cax)
    # cbar = plt.colorbar()
    # cbar.set_label('In-plane displacement ($\AA$)',rotation=90, fontsize = 20, fontweight = 'bold', labelpad=20)
    # cbar.ax.tick_params(labelsize = 18)
    # plt.xticks(fontsize = 18)
    # plt.yticks(fontsize = 18)
    # fig = plt.gcf()
    # fig.set_size_inches(7.5, 10)
    # plt.tight_layout()
    # fig.savefig(figname, dpi=200)
    # plt.show()


def plot_energy_side_by_side(twist_angle, pot1, pot2, label1, label2, atom_type=1):
    d1 = get_data(twist_angle, pot1)
    d2 = get_data(twist_angle, pot2)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6, 5))
    scatterplotter(fig, ax1, d1.x, d1.y, d1.energy, label1, ylabel='$y$ (ang)', colorbar=False)
    scatterplotter(fig, ax2, d2.x, d2.y, d2.energy, label2, ylabel=None, colorbar=True)
    fig.tight_layout()
    plt.savefig(f'{twist_angle}_energy.pdf', bbox_inches='tight')


def plot_displacement_side_by_side(twist_angle, pot1, pot2, label1, label2, atom_type=1):
    d1 = get_disp_data(twist_angle, pot1)
    d2 = get_disp_data(twist_angle, pot2)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6, 5))
    quiverplotter(fig, ax1, d1.x, d1.y, d1.dx, d1.dy, d1.mag, label1, ylabel='$y$ (ang)', colorbar=False)
    quiverplotter(fig, ax2, d2.x, d2.y, d2.dx, d2.dy, d2.mag, label2, ylabel=None, colorbar=True)
    fig.tight_layout()
    plt.savefig(f'{twist_angle}_disp.pdf', bbox_inches='tight')

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
        # d_ouyang = plot_energy(twist_angle, 'ouyang', '(a) Ouyang', strip_width=strip_width)
        # d_refit = plot_energy(twist_angle, 'refit', '(b) QMC', strip_width=strip_width)
        # plot_energy_side_by_side(twist_angle, 'ouyang', 'refit', '(a) Ouyang', '(b) QMC', atom_type=1)
        plot_displacement_side_by_side(twist_angle, 'ouyang', 'refit', '(a) Ouyang', '(b) QMC', atom_type=1)
        # d = pd.concat([d_ouyang, d_refit], ignore_index=True)
        # d_strip = d.loc[(-strip_width <= d.x) & (d.x < strip_width), :]
        # print(d_strip.shape)
        # plot_energy_strip(d_strip, f'{twist_angle}_energy_strip.pdf')
