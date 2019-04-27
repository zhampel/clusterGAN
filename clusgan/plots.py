from __future__ import print_function

try:
    import os
    import numpy as np

    import matplotlib
    import matplotlib.pyplot as plt
    
except ImportError as e:
    print(e)
    raise ImportError


def plot_train_loss(df=[], arr_list=[''], figname='training_loss.png'):

    fig, ax = plt.subplots(figsize=(16,10))
    for arr in arr_list:
        label = df[arr][0]
        vals = df[arr][1]
        epochs = range(0, len(vals))
        ax.plot(epochs, vals, label=r'%s'%(label))
    
    ax.set_xlabel('Epoch', fontsize=18)
    ax.set_ylabel('Loss', fontsize=18)
    ax.set_title('Training Loss', fontsize=24)
    ax.grid()
    #plt.yscale('log')
    plt.legend(loc='upper right', numpoints=1, fontsize=16)
    print(figname)
    plt.tight_layout()
    fig.savefig(figname)
