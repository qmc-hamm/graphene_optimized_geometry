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
    # theta = (float(twist_angle.replace('-', '.'))/2) *np.pi/180
    theta = 0
    d_rotated = d.copy()
    d_rotated[['x', 'y', 'z']] = rotate_coords(d[['x', 'y', 'z']], theta)
    d_rotated['potential'] = potential

    tile1 = d_rotated.copy()
    tile1['x'] -= latvec[0][0]

    tile2 = d_rotated.copy()
    tile2['y'] += latvec[1][1]

    tile3 = d_rotated.copy()
    tile3['x'] -= latvec[0][0]
    tile3['y'] += latvec[1][1]
    d = pd.concat([tile1, tile2, tile3, d_rotated], ignore_index=True)
    return d

def get_disp_data(twist_angle, pot, atom_type=1):
    '''
    Subtracts to initial coordinates from
    '''
    d_final = get_data(twist_angle, pot, relaxed=True, atom_type=atom_type)
    d_init = get_data(twist_angle, pot, relaxed=False, atom_type=atom_type)
    d_init['dx'] = d_final['x'] - d_init['x']
    d_init['dy'] = d_final['y'] - d_init['y']
    d_init['dz'] = d_final['z'] - d_init['z']
    d_init['mag'] = (d_init['dx']**2 + d_init['dy']**2)**0.5
    return d_init

def scatterplotter(fig, ax, x, y, z, title, zlabel, vmin=None, vmax=None, ylabel=None, colorbar=True, strip_width=None):
    p = ax.scatter(x, y, c=z, cmap='jet', s=2, lw=0, vmin=vmin, vmax=vmax)
    ax.text(0.1, 1.02, title, transform=ax.transAxes)
    if strip_width is not None:
        ax.axvspan(0, strip_width, alpha=0.5, color='#2f2f2f')

    ax.set_aspect('equal', 'box')
    ax.set(
        xlabel='$x$ ($\\mathrm{\\AA}$)', xlim=(-20, 150),
        ylabel=ylabel, ylim=(None, 250)
        )
    vmin = min(z) if vmin is None else vmin
    vmax = max(z) if vmax is None else vmax

    if colorbar:
        cax = fig.add_axes([
            ax.get_position().x1 + 0.11, # x-coordinate corner
            ax.get_position().y0 - 0.02, # y-coordinate corner
            0.01, # width
            ax.get_position().height*1.13 # height
            ])
        cax.set_title(zlabel)
        ax.figure.colorbar(p, cax=cax)

def quiverplotter(fig, ax, x, y, dx, dy, mag, title, zlabel, ylabel=None, colorbar=True):
    p = ax.quiver(x, y, dx, dy, mag, cmap='jet', headlength=20, headwidth=24, headaxislength=9)
    ax.text(0.1, 1.02, title, transform=ax.transAxes)
    ax.set_aspect('equal', 'box')
    ax.set(
        xlabel='$x$ ($\\mathrm{\\AA}$)', xlim=(0, 140),
        ylabel=ylabel, ylim=(0, 248)
        )
    if colorbar:
        cax = fig.add_axes([
            ax.get_position().x1 + 0.11, # x-coordinate corner
            ax.get_position().y0 - 0.02, # y-coordinate corner
            0.01, # width
            ax.get_position().height*1.13 # height
            ])
        cax.set_title(zlabel)
        ax.figure.colorbar(p, cax=cax)

def plot_energy_side_by_side(twist_angle, pot1, pot2, label1, label2, atom_type=1):
    d1 = get_data(twist_angle, pot1, atom_type=atom_type)
    d2 = get_data(twist_angle, pot2, atom_type=atom_type)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6, 5))
    scatterplotter(fig, ax1, d1.x, d1.y, d1.energy, label1, 'Energy (eV/atom)', vmin=-7.427, vmax=-7.412, ylabel='$y$ ($\\mathrm{\\AA}$)', colorbar=False)
    scatterplotter(fig, ax2, d2.x, d2.y, d2.energy, label2, 'Energy (eV/atom)', vmin=-7.427, vmax=-7.412, ylabel=None, colorbar=True)
    fig.tight_layout()
    plt.savefig(f'{twist_angle}_energy.pdf', bbox_inches='tight')

def plot_z_side_by_side(twist_angle, pot1, pot2, label1, label2, atom_type=1):
    d1 = get_data(twist_angle, pot1, atom_type=atom_type)
    d2 = get_data(twist_angle, pot2, atom_type=atom_type)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6, 5))
    scatterplotter(fig, ax1, d1.x, d1.y, d1.z, label1, '$z$ ($\\mathrm{\\AA}$)', vmin=2.9, vmax=3.1, ylabel='$y$ ($\\mathrm{\\AA}$)', colorbar=False)
    scatterplotter(fig, ax2, d2.x, d2.y, d2.z, label2, '$z$ (ng)', vmin=2.9, vmax=3.1, ylabel=None, colorbar=True)
    fig.tight_layout()
    plt.savefig(f'{twist_angle}_z.pdf', bbox_inches='tight')

def plot_displacement_vectors_side_by_side(twist_angle, pot1, pot2, label1, label2, atom_type=1):
    d1 = get_disp_data(twist_angle, pot1, atom_type=atom_type)
    d2 = get_disp_data(twist_angle, pot2, atom_type=atom_type)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6, 5))
    quiverplotter(fig, ax1, d1.x, d1.y, d1.dx, d1.dy, d1.mag, label1, '$r$ ($\\mathrm{\\AA}$)', ylabel='$y$ ($\\mathrm{\\AA}$)', colorbar=False)
    quiverplotter(fig, ax2, d2.x, d2.y, d2.dx, d2.dy, d2.mag, label2, '$r$ ($\\mathrm{\\AA}$)', ylabel=None, colorbar=True)
    fig.tight_layout()
    plt.savefig(f'{twist_angle}_disp.pdf', bbox_inches='tight')


def plot_displacement_magnitude_side_by_side(twist_angle, pot1, pot2, label1, label2, atom_type=1, strip=None):
    d1 = get_disp_data(twist_angle, pot1, atom_type=atom_type)
    d2 = get_disp_data(twist_angle, pot2, atom_type=atom_type)
    if strip is not None:
        d1 = d1.loc[(strip[0] <= d1.x) & (d1.x < strip[1]), :]
        # d1['x'] += 20
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6, 5))
    scatterplotter(fig, ax1, d1.x, d1.y, d1.mag, label1, '$r$ ($\\mathrm{\\AA}$)', vmin=0, vmax=None, ylabel='$y$ ($\\mathrm{\\AA}$)', colorbar=False)
    scatterplotter(fig, ax2, d2.x, d2.y, d2.mag, label2, '$r$ ($\\mathrm{\\AA}$)', vmin=0, vmax=None, ylabel=None, colorbar=True)
    fig.tight_layout()
    plt.savefig(f'{twist_angle}_mag.pdf', bbox_inches='tight', dpi=600)

def find_sp_peaks(x, y):
    peaks, _ = find_peaks(y)
    smallest_two = sorted(y[peaks])[:2]
    peak_idx1, peak_idx2 = np.where((y == smallest_two[0]) | (y == smallest_two[1]))[0]
    return x[peak_idx1], x[peak_idx2]

def lineplotter(x, y_ouyang, y_refit, xlabel, ylabel, output):
    x_ouyang1, x_ouyang2 = find_sp_peaks(x, y_ouyang)
    x_refit1, x_refit2 = find_sp_peaks(x, y_refit)

    width_ouyang = x_ouyang2 - x_ouyang1
    width_refit = x_refit2 - x_refit1

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
    colors = sns.color_palette()

    ls = '-'
    ax.plot(x, y_refit, ls, color=colors[0],
        label=f"KC-QMC: $W_\\mathrm{{D}}$ = {width_refit:.1f} " + "$\\mathrm{\\AA}$"
        )
    ax.plot(x, y_ouyang, ls, color=colors[1],
        label=f"KC-Ouyang: $W_\\mathrm{{D}}$ = {width_ouyang:.1f} " + "$\\mathrm{\\AA}$"
        )

    offset = 8
    # ax.annotate(text='', xy=(x_refit2+offset, 0.06), xytext=(x_refit1-offset, 0.06), arrowprops=dict(arrowstyle='<->'))
    # ax.text(108, 0.07, '$W_\\mathrm{{D}}$')

    ax.set(ylim=(0.0, 0.3))
    ax.legend(loc='upper right', frameon=False, edgecolor='k', fontsize=8)
    ax.set(xlabel=xlabel, xlim=(None, None),
        ylabel=ylabel)
    fig.tight_layout()
    plt.savefig(output, bbox_inches='tight', dpi=600)
    plt.close()

    return width_ouyang, width_refit

def plot_displacement_magnitude_1d(twist_angle, pot1, pot2, label1, label2, strip, atom_type=1):
    d1 = get_disp_data(twist_angle, pot1, atom_type=atom_type)
    d2 = get_disp_data(twist_angle, pot2, atom_type=atom_type)

    d1 = d1.loc[(strip[0] <= d1.x) & (d1.x < strip[1]), :]
    d2 = d2.loc[(strip[0] <= d2.x) & (d2.x < strip[1]), :]

    d1 = d1.sort_values(by='y')
    d2 = d2.sort_values(by='y')

    spline1 = make_interp_spline(d1.y, d1.mag)
    spline2 = make_interp_spline(d2.y, d2.mag)
    lin = np.linspace(0, 250, 1000)

    width_ouyang, width_refit = lineplotter(lin, spline1(lin), spline2(lin), 'Distance along the line ($\\mathrm{\\AA}$)', 'In-plane displacement magnitude ($\\mathrm{\\AA}$)', f'{twist_angle}_mag_1d.pdf')
    # width_ouyang, width_refit = lineplotter(d1.y, d1.mag, d2.mag, 'Distance along the line ($\\mathrm{\\AA}$)', 'In-plane displacement magnitude ($\\mathrm{\\AA}$)', f'{twist_angle}_mag_1d.pdf')

    print(width_ouyang, width_refit)

if __name__ == '__main__':
    for twist_angle in ['0-99']:
        # plot_energy_side_by_side(twist_angle, 'ouyang', 'refit', '(a) KC-Ouyang', '(b) KC-QMC')
        # plot_z_side_by_side(twist_angle, 'ouyang', 'refit', '(a) KC-Ouyang', '(b) KC-QMC')
        # plot_displacement_vectors_side_by_side(twist_angle, 'ouyang', 'refit', '(a) KC-Ouyang', '(b) KC-QMC')
        strip = (-0.7, 0.7)
        # strip = (-0.5, 1.3)
        plot_displacement_magnitude_side_by_side(twist_angle, 'ouyang', 'refit', '(a) KC-Ouyang', '(b) KC-QMC', strip=strip)
        plot_displacement_magnitude_1d(twist_angle, 'ouyang', 'refit', '(a) KC-Ouyang', '(b) KC-QMC', strip)
