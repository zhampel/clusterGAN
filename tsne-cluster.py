from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np
    import sys
    np.set_printoptions(threshold=sys.maxsize)

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    import pandas as pd
    
    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    
    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    
    from itertools import chain as ichain

    from clusgan.definitions import DATASETS_DIR, RUNS_DIR
    from clusgan.models import Generator_CNN, Encoder_CNN, Discriminator_CNN
    from clusgan.datasets import get_dataloader, dataset_list

    from sklearn.manifold import TSNE
except ImportError as e:
    print(e)
    raise ImportError


def main():
    global args
    parser = argparse.ArgumentParser(description="TSNE generation script")
    parser.add_argument("-r", "--run_dir", dest="run_dir", help="Training run directory")
    parser.add_argument("-p", "--perplexity", dest="perplexity", default=-1, type=int,  help="TSNE perplexity")
    parser.add_argument("-n", "--n_samples", dest="n_samples", default=100, type=int,  help="Number of samples")
    args = parser.parse_args()

    # TSNE setup
    n_samples = args.n_samples
    perplexity = args.perplexity
    
    # Directory structure for this run
    run_dir = args.run_dir.rstrip("/")
    run_name = run_dir.split(os.sep)[-1]
    dataset_name = run_dir.split(os.sep)[-2]
    
    run_dir = os.path.join(RUNS_DIR, dataset_name, run_name)
    data_dir = os.path.join(DATASETS_DIR, dataset_name)
    imgs_dir = os.path.join(run_dir, 'images')
    models_dir = os.path.join(run_dir, 'models')


    # Latent space info
    train_df = pd.read_csv('%s/training_details.csv'%(run_dir))
    latent_dim = train_df['latent_dim'][0]
    n_c = train_df['n_classes'][0]

    cuda = True if torch.cuda.is_available() else False
    
    # Load encoder model
    encoder = Encoder_CNN(latent_dim, n_c)
    enc_figname = os.path.join(models_dir, encoder.name + '.pth.tar')
    encoder.load_state_dict(torch.load(enc_figname))
    encoder.cuda()
    encoder.eval()

    # Configure data loader
    dataloader = get_dataloader(dataset_name=dataset_name, data_dir=data_dir, batch_size=n_samples, train_set=False)
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Load TSNE
    if (perplexity < 0):
        tsne = TSNE(n_components=2, verbose=1, init='pca', random_state=0)
        fig_title = "PCA Initialization"
        figname = os.path.join(run_dir, 'tsne-pca.png')
    else:
        tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=300)
        fig_title = "Perplexity = $%d$"%perplexity
        figname = os.path.join(run_dir, 'tsne-plex%i.png'%perplexity)

    # Get full batch for encoding
    imgs, labels = next(iter(dataloader))
    c_imgs = Variable(imgs.type(Tensor), requires_grad=False)
    
    # Encode real images
    enc_zn, enc_zc, enc_zc_logits = encoder(c_imgs)
    # Stack latent space encoding
    enc = np.hstack((enc_zn.cpu().detach().numpy(), enc_zc_logits.cpu().detach().numpy()))
    #enc = np.hstack((enc_zn.cpu().detach().numpy(), enc_zc.cpu().detach().numpy()))

    # Cluster with TSNE
    tsne_enc = tsne.fit_transform(enc)

    # Convert to numpy for indexing purposes
    labels = labels.cpu().data.numpy()

    # Color and marker for each true class
    colors = cm.rainbow(np.linspace(0, 1, n_c))
    markers = matplotlib.markers.MarkerStyle.filled_markers

    # Save TSNE figure to file
    fig, ax = plt.subplots(figsize=(16,10))
    for iclass in range(0, n_c):
        # Get indices for each class
        idxs = labels==iclass
        # Scatter those points in tsne dims
        ax.scatter(tsne_enc[idxs, 0],
                   tsne_enc[idxs, 1],
                   marker=markers[iclass],
                   c=colors[iclass],
                   edgecolor=None,
                   label=r'$%i$'%iclass)

    ax.set_title(r'%s'%fig_title, fontsize=24)
    ax.set_xlabel(r'$X^{\mathrm{tSNE}}_1$', fontsize=18)
    ax.set_ylabel(r'$X^{\mathrm{tSNE}}_2$', fontsize=18)
    plt.legend(title=r'Class', loc='best', numpoints=1, fontsize=16)
    plt.tight_layout()
    fig.savefig(figname)

if __name__ == "__main__":
    main()
