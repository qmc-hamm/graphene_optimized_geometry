import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

def scatterplotter(x, y, z, caption, output):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 5))
    p = plt.scatter(x, y, c=z, cmap='jet', s=2, alpha=0.8, lw=0)

    cax = fig.add_axes([
        ax.get_position().x1+0.01 + 0.1,
        ax.get_position().y0 + 0.05,
        0.02,
        ax.get_position().height
        ])
    cbar = plt.colorbar(p, cax=cax)
    cbar.set_label(caption, rotation=90, fontsize=9, fontweight='bold', labelpad=8)
    cbar.ax.tick_params(labelsize=9)
    plt.clim(vmin=np.min(z), vmax=np.max(z))
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    ax.set_aspect('equal', 'box')
    ax.set(xlabel='$x$ (ang)', ylabel='$y$ (ang)')
    fig.tight_layout()
    plt.savefig(output, bbox_inches='tight')
    plt.close()

def lineplotter(x, y, xlabel, ylabel, output):
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(y)
    smallest_two = sorted(y[peaks])[:2]
    peak_idx1, peak_idx2 = np.where((y == smallest_two[0]) | (y == smallest_two[1]))[0]
    width = x[peak_idx2] - x[peak_idx1]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
    ax.plot(x, y, '-')
    colors = sns.color_palette()
    for idx in [peak_idx1, peak_idx2]:
        ax.axvline(x=x[idx], c=colors[0], ls='--')
    ax.text((x[peak_idx2] + x[peak_idx2])/2, smallest_two[0], f'$W_\\mathrm{{D}}$ = {width:.1f} ang')
    ax.set(xlabel=xlabel, ylabel=ylabel)
    fig.tight_layout()
    plt.savefig(output, bbox_inches='tight')
    plt.close()

def plot_dislocation(twist_angle, potential, atom_type=1):

    d1, latvec1 = dumpfile_to_df(f"kc_rebo_{potential}/raw/simulations/{twist_angle}/dump_initial.txt")
    d2, latvec2 = dumpfile_to_df(f"kc_rebo_{potential}/raw/simulations/{twist_angle}/dump_final.txt")

    d1_layer = d1.loc[d1.atom_type == atom_type, :]
    d2_layer = d2.loc[d2.atom_type == atom_type, :]

    dx_bot = d2_layer['x'] - d1_layer['x']
    dy_bot = d2_layer['y'] - d2_layer['y']

    d1_layer['magnitude'] = (dx_bot**2 + dy_bot**2)**0.5
    d1_layer_1d = d1_layer.loc[(0 <= d1_layer['x']) & (d1_layer['x'] <= 1.3), :]
    d1_layer_1d = d1_layer_1d.sort_values(by=['y'])

    from scipy.interpolate import make_interp_spline
    spline = make_interp_spline(d1_layer_1d.y, d1_layer_1d.magnitude)
    lin = np.linspace(0, np.max(d1_layer_1d.y), 1000)


    lineplotter(lin, spline(lin), 'Distance along the line (ang)', 'In-plane displacement magnitude (ang)', f'{twist_angle}_{potential}_mag_1d.pdf')
    scatterplotter(d1_layer.x, d1_layer.y, d1_layer.magnitude, 'In-plane displacement magnitude (ang)', f'{twist_angle}_{potential}_mag.pdf')


def plot_energy(twist_angle, potential, atom_type=1):
    d1, latvec1 = dumpfile_to_df(f"kc_rebo_{potential}/raw/simulations/{twist_angle}/dump_initial.txt")
    d2, latvec2 = dumpfile_to_df(f"kc_rebo_{potential}/raw/simulations/{twist_angle}/dump_final.txt")

    d2_layer = d2.loc[d2.atom_type == atom_type, :]
    d2_layer_1d = d2_layer.loc[(0 <= d2_layer['x']) & (d2_layer['x'] <= 1.3), :]
    # d2_layer_1d = d2_layer_1d.sort_values(by=['y'])

    lineplotter(d2_layer_1d.y, d2_layer_1d.energy, 'Distance along the line (ang)', 'Energy (eV/atom)', f'{twist_angle}_{potential}_energy_1d.pdf')
    scatterplotter(d2_layer.x, d2_layer.y, d2_layer.energy, 'Energy (eV/atom)', f'{twist_angle}_{potential}_energy.pdf')


if __name__ == '__main__':
    for twist_angle in ['0-99']:
        for potential in ['ouyang', 'refit']:
            plot_dislocation(twist_angle, potential)
            # plot_energy(twist_angle, potential)
