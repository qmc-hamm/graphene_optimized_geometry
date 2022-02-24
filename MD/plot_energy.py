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
        xlabel='$x$ (ang)', xlim=(0, 140),
        ylabel=ylabel, ylim=(0, 248)
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
        cax.set_title(zlabel)
        ax.figure.colorbar(p, cax=cax)

def plot_energy_side_by_side(twist_angle, pot1, pot2, label1, label2, atom_type=1):
    d1 = get_data(twist_angle, pot1, atom_type=atom_type)
    d2 = get_data(twist_angle, pot2, atom_type=atom_type)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6, 5))
    scatterplotter(fig, ax1, d1.x, d1.y, d1.energy, label1, 'Energy (eV/atom)', vmin=-7.427, vmax=-7.412, ylabel='$y$ (ang)', colorbar=False)
    scatterplotter(fig, ax2, d2.x, d2.y, d2.energy, label2, 'Energy (eV/atom)', vmin=-7.427, vmax=-7.412, ylabel=None, colorbar=True)
    fig.tight_layout()
    plt.savefig(f'{twist_angle}_energy.pdf', bbox_inches='tight')

def plot_z_side_by_side(twist_angle, pot1, pot2, label1, label2, atom_type=1):
    d1 = get_data(twist_angle, pot1, atom_type=atom_type)
    d2 = get_data(twist_angle, pot2, atom_type=atom_type)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6, 5))
    scatterplotter(fig, ax1, d1.x, d1.y, d1.z, label1, '$z$ (ang)', vmin=2.9, vmax=3.1, ylabel='$y$ (ang)', colorbar=False)
    scatterplotter(fig, ax2, d2.x, d2.y, d2.z, label2, '$z$ (ng)', vmin=2.9, vmax=3.1, ylabel=None, colorbar=True)
    fig.tight_layout()
    plt.savefig(f'{twist_angle}_z.pdf', bbox_inches='tight')

def plot_displacement_vectors_side_by_side(twist_angle, pot1, pot2, label1, label2, atom_type=1):
    d1 = get_disp_data(twist_angle, pot1, atom_type=atom_type)
    d2 = get_disp_data(twist_angle, pot2, atom_type=atom_type)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6, 5))
    quiverplotter(fig, ax1, d1.x, d1.y, d1.dx, d1.dy, d1.mag, label1, '$r$ (ang)', ylabel='$y$ (ang)', colorbar=False)
    quiverplotter(fig, ax2, d2.x, d2.y, d2.dx, d2.dy, d2.mag, label2, '$r$ (ang)', ylabel=None, colorbar=True)
    fig.tight_layout()
    plt.savefig(f'{twist_angle}_disp.pdf', bbox_inches='tight')


def plot_displacement_magnitude_side_by_side(twist_angle, pot1, pot2, label1, label2, atom_type=1, strip_width=0.25):
    d1 = get_disp_data(twist_angle, pot1, atom_type=atom_type)
    d2 = get_disp_data(twist_angle, pot2, atom_type=atom_type)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6, 5))
    scatterplotter(fig, ax1, d1.x, d1.y, d1.mag, label1, '$r$', vmin=0, vmax=None, ylabel='$y$ (ang)', colorbar=False, strip_width=strip_width)
    scatterplotter(fig, ax2, d2.x, d2.y, d2.mag, label2, '$r$', vmin=0, vmax=None, ylabel=None, colorbar=True, strip_width=strip_width)
    fig.tight_layout()
    plt.savefig(f'{twist_angle}_mag.pdf', bbox_inches='tight')

def find_sp_peaks(x, y):
    peaks, _ = find_peaks(y)
    smallest_two = sorted(y[peaks])[:2]
    peak_idx1, peak_idx2 = np.where((y == smallest_two[0]) | (y == smallest_two[1]))[0]
    return x[peak_idx1], x[peak_idx2]

def lineplotter(x, y_ouyang, y_refit, xlabel, ylabel, output, plot=True):
    x_ouyang1, x_ouyang2 = find_sp_peaks(x, y_ouyang)
    x_refit1, x_refit2 = find_sp_peaks(x, y_refit)

    width_ouyang = x_ouyang2 - x_ouyang1
    width_refit = x_refit2 - x_refit1

    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
        colors = sns.color_palette()

        ax.plot(x, y_refit, '-', color=colors[0], label=f'QMC: $W_\\mathrm{{D}}$ = {width_refit:.1f} ang')
        for peak_x in [x_refit1, x_refit2]:
            ax.axvline(x=peak_x, c=colors[0], ls='--')

        ax.plot(x, y_ouyang, '-', color=colors[1], label=f'Ouyang: $W_\\mathrm{{D}}$ = {width_ouyang:.1f} ang')
        for peak_x in [x_ouyang1, x_ouyang2]:
            ax.axvline(x=peak_x, c=colors[1], ls='--')

        ax.set(ylim=(None, 0.25))
        ax.legend(loc='upper right', fancybox=False, edgecolor='k')
        ax.set(xlabel=xlabel, ylabel=ylabel)
        fig.tight_layout()
        plt.savefig(output, bbox_inches='tight')
        plt.close()

    return width_ouyang, width_refit

# def plot_energy_strip(d, output):
    # d = pd.concat([d_ouyang, d_refit], ignore_index=True)
        # d_strip = d.loc[(-strip_width <= d.x) & (d.x < strip_width), :]
        # print(d_strip.shape)

#     g = sns.FacetGrid(data=d, hue='potential')
#     colors = sns.color_palette()
#     for i, potential in enumerate(g.hue_names):
#         dd = d.loc[d.potential == potential, :]
#         g.ax.plot(dd.y, dd.energy, 'o-', label=potential, color=colors[i], ms=3, lw=1, mec='white', mew=0.1)
#     g.ax.legend(bbox_to_anchor=(0.45, 0.72), frameon=False, fontsize=8)
#     g.set(
#         xlabel='Distance along the $y$-axis ($\\mathrm{\\AA}$)',
#         ylabel='Energy (eV/atom)'
#         )
#     g.fig.set_size_inches(3, 3)
#     plt.savefig(output, bbox_inches='tight')

def plot_displacement_magnitude_1d(twist_angle, pot1, pot2, label1, label2, atom_type=1, strip_width=0.25):
    d1 = get_disp_data(twist_angle, pot1, atom_type=atom_type)
    d2 = get_disp_data(twist_angle, pot2, atom_type=atom_type)
    # d = pd.concat([d1, d2], ignore_index=True)

    # d = d.loc[(-strip_width <= d.x) & (d.x < strip_width), :]
    d1 = d1.sort_values(by='y')
    d2 = d2.sort_values(by='y')
    print(d1)
    spline_ouyang = make_interp_spline(d1.y, d1.mag)
    spline_refit = make_interp_spline(d2.y, d2.mag)
    lin = np.linspace(0, np.max(d1.y), 1000)

    width_ouyang, width_refit = lineplotter(lin, spline_ouyang(lin), spline_refit(lin), 'Distance along the line (ang)', 'In-plane displacement magnitude (ang)', f'{twist_angle}_mag_1d.pdf')
    # g = sns.FacetGrid(data=d, hue='potential')
    # colors = sns.color_palette()
    # for i, potential in enumerate(g.hue_names):
    #     dd = d.loc[d.potential == potential, :]
    #     g.ax.plot(dd.y, dd.mag, 'o-', label=potential, color=colors[i], ms=3, lw=1, mec='white', mew=0.1)
    # g.ax.legend(bbox_to_anchor=(0.45, 0.72), frameon=False, fontsize=8)
    # g.set(
    #     xlabel='Distance along the $y$-axis ($\\mathrm{\\AA}$)',
    #     ylabel='Energy (eV/atom)'
    #     )
    # g.fig.set_size_inches(3, 3)
    # plt.savefig(f'{twist_angle}_mag_1d.pdf', bbox_inches='tight')
    # plt.close()
    print(width_ouyang, width_refit)

if __name__ == '__main__':
    widths = []
    for twist_angle in ['0-99']:
        plot_energy_side_by_side(twist_angle, 'ouyang', 'refit', '(a) Ouyang', '(b) QMC', atom_type=1)
        plot_z_side_by_side(twist_angle, 'ouyang', 'refit', '(a) Ouyang', '(b) QMC', atom_type=1)
        plot_displacement_vectors_side_by_side(twist_angle, 'ouyang', 'refit', '(a) Ouyang', '(b) QMC', atom_type=1)
        plot_displacement_magnitude_side_by_side(twist_angle, 'ouyang', 'refit', '(a) Ouyang', '(b) QMC', atom_type=1, strip_width=1.3)
        # plot_displacement_magnitude_1d(twist_angle, 'ouyang', 'refit', '(a) Ouyang', '(b) QMC', atom_type=1, strip_width=1.3)
