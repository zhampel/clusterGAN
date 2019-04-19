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
    from clusgan.utils import sample_z

    from sklearn.manifold import TSNE
except ImportError as e:
    print(e)
    raise ImportError

def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default='clusgan', help="Name of training run")
    args = parser.parse_args()

    n_sne = 100
    batch_size = n_sne
    
    # Make directory structure for this run
    run_name = args.run_name
    run_dir = '%s/%s'%(RUNS_DIR, run_name)
    imgs_dir = '%s/images'%(run_dir)


    # Latent space info
    latent_dim = 30
    n_c = 10

    cuda = True if torch.cuda.is_available() else False
    
    # Load encoder model
    encoder = Encoder_CNN(latent_dim, n_c)
    enc_fname = "%s/encoder_MNIST.pth.tar"%(run_dir)
    encoder.load_state_dict(torch.load(enc_fname))
    encoder.cuda()

    # Load encoder model
    x_shape = (1, 28, 28)
    generator = Generator_CNN(latent_dim, n_c, x_shape)
    gen_fname = "%s/generator_MNIST.pth.tar"%(run_dir)
    generator.load_state_dict(torch.load(gen_fname))
    generator.cuda()
    generator.eval()
    
    for idx in range(n_c):
        zn, zc, zc_idx = sample_z(shape=batch_size, latent_dim=latent_dim, n_c=n_c, fix_class=idx, req_grad=False)
    
        # Generate a batch of images
        gen_imgs = generator(zn, zc)
        
        # Save some examples!
        save_image(gen_imgs.data, '%s/class_%i_gen.png' %(imgs_dir, idx), 
                   nrow=int(np.sqrt(batch_size)), normalize=True)

        enc_zn, enc_zc, enc_zc_logits = encoder(gen_imgs)
        print(idx)
        print(zc_idx)
        print(torch.argmax(enc_zc, dim=1))
        
        # Generate a batch of images
        gen_imgs = generator(enc_zn, enc_zc)
        
        # Save some examples!
        save_image(gen_imgs.data, '%s/class_enc_%i_gen.png' %(imgs_dir, idx), 
                   nrow=int(np.sqrt(batch_size)), normalize=True)
        print(torch.argmax(enc_zc, dim=1))
        enc_zn, enc_zc, enc_zc_logits = encoder(gen_imgs)
        print(torch.argmax(enc_zc, dim=1))

            

    #Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    #perplexity = 40
    #tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=2000)

    ## Get full batch for encoding
    #imgs, labels = next(iter(dataloader))
    #c_imgs = Variable(imgs.type(Tensor), requires_grad=False)
    #enc_zn, enc_zc, enc_zc_logits = encoder(c_imgs)

    ## Stack latent space encoding
    #enc = np.hstack((enc_zn.cpu().detach().numpy(), enc_zc.cpu().detach().numpy()))

    ## Cluster with TSNE
    #tsne_enc = tsne.fit_transform(enc)

    #fname = '%s/tsne.png'%(run_dir)
    #fig, ax = plt.subplots(figsize=(16,10))
    #ax.set_title("Perplexity=%d" % perplexity)
    #ax.scatter(tsne_enc[:, 0], tsne_enc[:, 1], c=labels)
    #ax.axis('tight')
    #fig.savefig(fname)

if __name__ == "__main__":
    main()
