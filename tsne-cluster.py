from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np
    import sys
    np.set_printoptions(threshold=sys.maxsize)

    import matplotlib
    import matplotlib.pyplot as plt

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
    parser.add_argument("-r", "--run_name", dest="run_name", default='clusgan', help="Name of training run")
    parser.add_argument("-s", "--dataset_name", dest="dataset_name", default='mnist', choices=dataset_list,  help="Dataset name")
    parser.add_argument("-p", "--perplexity", dest="perplexity", default=40, type=int,  help="TSNE perplexity")
    args = parser.parse_args()

    # TSNE setup
    n_sne = 10000
    batch_size = n_sne
    perplexity = args.perplexity
    
    # Directory structure for this run
    run_name = args.run_name
    dataset_name = args.dataset_name
    
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
    enc_fname = os.path.join(models_dir, encoder.name + '.pth.tar')
    encoder.load_state_dict(torch.load(enc_fname))
    encoder.cuda()
    encoder.eval()

    # Configure data loader
    dataloader = get_dataloader(dataset_name=dataset_name, data_dir=data_dir, batch_size=batch_size, train_set=False)
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Load TSNE
    tsne = TSNE(n_components=2, verbose=1, init='pca', random_state=0)
    #tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=300)

    # Get full batch for encoding
    imgs, labels = next(iter(dataloader))
    c_imgs = Variable(imgs.type(Tensor), requires_grad=False)
    
    # Encode real images
    enc_zn, enc_zc, enc_zc_logits = encoder(c_imgs)
    # Stack latent space encoding
    enc = np.hstack((enc_zn.cpu().detach().numpy(), enc_zc.cpu().detach().numpy()))

    # Cluster with TSNE
    tsne_enc = tsne.fit_transform(enc)

    # Save TSNE figure to file
    fname = os.path.join(run_dir, 'tsne.png')
    fig, ax = plt.subplots(figsize=(16,10))
    ax.set_title("Perplexity = %d" % perplexity)
    ax.scatter(tsne_enc[:, 0], tsne_enc[:, 1], c=labels)
    ax.axis('tight')
    fig.savefig(fname)

if __name__ == "__main__":
    main()
