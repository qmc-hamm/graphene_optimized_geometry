import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rc('font', family='serif')
# plt.rc('text', usetex=True)

if __name__ == '__main__':
    d = pd.read_csv('width.csv')
    print(d)
    d['label'] = d['potential'].map({'ouyang': 'Ouyang', 'refit': 'QMC'})

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
    s = sns.lineplot(data=d, x='twist_angle', y='width', hue='label', marker='o')
    s.set(xlabel='Twist angle (degrees)', ylabel='$W_\\mathrm{{D}}$ (ang)')
    fig.tight_layout()
    plt.savefig('width.pdf', bbox_inches='tight')
