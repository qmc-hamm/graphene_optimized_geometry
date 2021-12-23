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

def scatterplotter(x, y, z, title, colorbar_label, output, stripe=False):
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
    if stripe:
        ax.axvspan(0, 1.3, alpha=0.5, color='#2f2f2f')
    plt.clim(vmin=np.min(z), vmax=np.max(z))
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    ax.set_aspect('equal', 'box')
    ax.set(xlabel='$x$ (ang)', ylabel='$y$ (ang)')
    fig.tight_layout()
    plt.savefig(output, bbox_inches='tight')
    plt.close()

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

        ax.plot(x, y_refit, '-', color=colors[0], label=f'Refit: $W_\\mathrm{{D}}$ = {width_refit:.1f} ang')
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

def process_dislocation(twist_angle, potential, title, atom_type=1):

    d1, latvec1 = dumpfile_to_df(f"kc_rebo_{potential}/raw/simulations/{twist_angle}/dump_initial.txt")
    d2, latvec2 = dumpfile_to_df(f"kc_rebo_{potential}/raw/simulations/{twist_angle}/dump_final.txt")

    d1_layer = d1.loc[d1.atom_type == atom_type, :]
    d2_layer = d2.loc[d2.atom_type == atom_type, :]

    dx_bot = d2_layer['x'] - d1_layer['x']
    dy_bot = d2_layer['y'] - d2_layer['y']

    d1_layer['magnitude'] = (dx_bot**2 + dy_bot**2)**0.5
    scatterplotter(d1_layer.x, d1_layer.y, d1_layer.magnitude, title, 'In-plane displacement magnitude (ang)', f'{twist_angle}_{potential}_mag.pdf', stripe=True)

    d1_layer_1d = d1_layer.loc[(0 <= d1_layer['x']) & (d1_layer['x'] <= 1.3), :]
    d1_layer_1d = d1_layer_1d.sort_values(by=['y'])
    return d1_layer_1d

def plot_dislocation(twist_angle):
    d1_layer_1d_ouyang = process_dislocation(twist_angle, 'ouyang', '(a) Ouyang')
    d1_layer_1d_refit = process_dislocation(twist_angle, 'refit', '(b) Refit')

    spline_ouyang = make_interp_spline(d1_layer_1d_ouyang.y, d1_layer_1d_ouyang.magnitude)
    spline_refit = make_interp_spline(d1_layer_1d_refit.y, d1_layer_1d_refit.magnitude)
    lin = np.linspace(0, np.max(d1_layer_1d_ouyang.y), 1000)

    width_ouyang, width_refit = lineplotter(lin, spline_ouyang(lin), spline_refit(lin), 'Distance along the line (ang)', 'In-plane displacement magnitude (ang)', f'{twist_angle}_mag_1d.pdf')
    return width_ouyang, width_refit

def plot_energy(twist_angle, potential, title, atom_type=1):
    d1, latvec1 = dumpfile_to_df(f"kc_rebo_{potential}/raw/simulations/{twist_angle}/dump_initial.txt")
    d2, latvec2 = dumpfile_to_df(f"kc_rebo_{potential}/raw/simulations/{twist_angle}/dump_final.txt")

    d2_layer = d2.loc[d2.atom_type == atom_type, :]
    d2_layer_1d = d2_layer.loc[(0 <= d2_layer['x']) & (d2_layer['x'] <= 1.3), :]
    # d2_layer_1d = d2_layer_1d.sort_values(by=['y'])

    # lineplotter(d2_layer_1d.y, d2_layer_1d.energy, 'Distance along the line (ang)', 'Energy (eV/atom)', f'{twist_angle}_{potential}_energy_1d.pdf')
    scatterplotter(d2_layer.x, d2_layer.y, d2_layer.energy, title, 'Energy (eV/atom)', f'{twist_angle}_{potential}_energy.pdf')


if __name__ == '__main__':
    widths = []
    for twist_angle in ['0-99', '1-05', '1-08', '1-16', '1-47', '2-0', '2-88', '3-89', '4-4', '5-1', '6-0']:
        width_ouyang, width_refit = plot_dislocation(twist_angle)
        widths.append([float(twist_angle.replace('-', '.')), width_ouyang, width_refit])
        plot_energy(twist_angle, 'ouyang', '(a) Ouyang')
        plot_energy(twist_angle, 'refit', '(b) Refit')

    d = pd.DataFrame(widths)
    d.columns = ['twist_angle', 'width_ouyang', 'width_refit']

    d_ouyang = d.copy(deep=True)
    d_ouyang['potential'] = 'ouyang'
    d_ouyang['width'] = d_ouyang['width_ouyang']

    d_refit = d.copy(deep=True)
    d_refit['potential'] = 'refit'
    d_refit['width'] = d_refit['width_refit']


    d = pd.concat([d_ouyang, d_refit], ignore_index=True)[['twist_angle', 'potential', 'width']]
    print(d)
    d.to_csv('width.csv', index=False)
