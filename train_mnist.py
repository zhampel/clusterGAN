from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np

    import matplotlib
    import matplotlib.pyplot as plt
    
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
    from clusgan.utils import tlog, softmax, initialize_weights, calc_gradient_penalty, sample_z
except ImportError as e:
    print(e)
    raise ImportError

def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default='clusgan', help="Name of training run")
    args = parser.parse_args()

    # Make a directory for this run
    run_name = args.run_name
    run_dir = '%s/%s'%(RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    n_epochs = 200
    batch_size = 64
    lr = 1e-4
    b1 = 0.5
    b2 = 0.9 #99
    decay = 2.5*1e-5
    drop_out = 0.2
    n_cpu = 1
    img_size = 28
    channels = 1
    sample_interval = 400
   
    # Latent space info
    latent_dim = 30 #100
    n_c = 10
    betan = 10
    betac = 10
   
    # Wasserstein metric flag
    wass_metric=True
    
    x_shape = (channels, img_size, img_size)
    
    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loss function
    bce_loss = torch.nn.BCELoss()
    xe_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    
    # Initialize generator and discriminator
    generator = Generator_CNN(latent_dim, n_c, x_shape)
    encoder = Encoder_CNN(latent_dim, n_c)
    discriminator = Discriminator_CNN(wass_metric=wass_metric)
    
    print(generator)
    print(encoder)
    print(discriminator)
    
    
    if cuda:
        generator.cuda()
        encoder.cuda()
        discriminator.cuda()
        bce_loss.cuda()
        xe_loss.cuda()
        mse_loss.cuda()
        
    
    # Configure data loader
    data_dir = '%s/mnist'%DATASETS_DIR
    os.makedirs(data_dir, exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=batch_size, shuffle=True)
    
    optimizer_GE = torch.optim.Adam(ichain(generator.parameters(), encoder.parameters()), 
                                    lr=lr, betas=(b1, b2), weight_decay=decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=decay)
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------
    a_acc = []
    n_acc = []
    g_l = []
    e_l = []
    ge_l = []
    d_l = []
    
    c_zn = []
    c_zc = []
    c_i = []
    
    
    #for epoch in range(n_epochs, 2*n_epochs):
    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            
            #scheduler_GE.batch_step()
            #scheduler_D.batch_step()
            
            generator.zero_grad()
            encoder.zero_grad()
            discriminator.zero_grad()
            
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
    
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
            
            # ---------------------------
            #  Train Generator + Encoder
            # ---------------------------
            
            optimizer_GE.zero_grad()
            
            # Sample random latent variables
            zn, zc, zc_idx = sample_z(shape=imgs.shape[0], latent_dim=latent_dim, n_c=n_c, req_grad=True)
    
            # Generate a batch of images
            gen_imgs = generator(zn, zc)
            
            # Discriminator output from real and generated samples
            D_gen = discriminator(gen_imgs)
            D_real = discriminator(real_imgs)
            
            if wass_metric:
                # Wasserstein GAN loss
                g_loss = torch.mean(D_gen)
            else:
                # Vanilla GAN loss
                g_loss = bce_loss(D_gen, valid)
            
            ## Generate a batch of latent vars
            #enc_z = encoder(real_imgs)
            
            if wass_metric:
                # Wasserstein GAN loss
                e_loss = torch.mean(D_real)
            else:
                # Vanilla GAN loss
                e_loss = bce_loss(D_real, valid)
    
    
            # Step for Generator & Encoder, 5x less than for discriminator
            if (i % 5 == 0):
    
                # Encode the generated images
                enc_gen_zn, enc_gen_zc = encoder(gen_imgs)
    
                # Calculate losses for z_n, z_c
                zn_loss = mse_loss(enc_gen_zn, zn)
                zc_loss = xe_loss(enc_gen_zc, zc_idx)
    
                # Check requested metric
                if wass_metric:
                    # Wasserstein GAN loss
                    ge_loss = torch.mean(D_gen) + betan * torch.mean(zn_loss) + betac * torch.mean(zc_loss)
                    #ge_loss = torch.mean(torch.mean(D_gen) + betan * zn_loss + betac * zc_loss)
                else:
                    # Vanilla GAN loss
                    ge_loss = -torch.mean(tlog(D_gen)) + betan * torch.mean(zn_loss) + betac * torch.mean(zc_loss)
                    #ge_loss = torch.mean(-tlog(D_gen) + betan * zn_loss + betac * zc_loss)
    
                ge_loss.backward(retain_graph=True)
                optimizer_GE.step()
    
            # ---------------------
            #  Train Discriminator
            # ---------------------
    
            optimizer_D.zero_grad()
    
            # Measure discriminator's ability to classify real from generated samples
            if wass_metric:
                # Wasserstein GAN loss
                d_loss = torch.mean(D_real) - torch.mean(D_gen)
                
                # Additional gradient penalty term
                gradient_penalty = calc_gradient_penalty(discriminator, real_imgs, gen_imgs)
                gradient_penalty.backward(retain_graph=True)
                
                d_loss += gradient_penalty
                
            else:
                # Vanilla GAN loss
                d_loss = -torch.mean(tlog(D_real)) - torch.mean(tlog(1 - D_gen))
    
            d_loss.backward()
            optimizer_D.step()
    
            #if (i % 5 == 0):
            #    d_loss.backward()
            #    optimizer_D.step()
    
            #print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [E loss: %f]" % (epoch, 
            #                                                    n_epochs, i, len(dataloader),
            #                                                    d_loss.item(), g_loss.item(), e_loss.item()))
    
            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                # Cycle through real -> enc -> gen
                r_imgs = real_imgs.data[:25]
                e_zn, e_zc = encoder(r_imgs, seval=True)
                reg_imgs = generator(e_zn, e_zc)
                img_mse_loss = mse_loss(r_imgs, reg_imgs)
                c_i.append(img_mse_loss.item())
                
                # Cycle through enc -> gen -> enc
                zn_samp, zc_samp, zc_samp_idx = sample_z(shape=25, latent_dim=latent_dim, n_c=n_c, req_grad=False)
                gen_imgs_samp = generator(zn_samp, zc_samp)
                zn_e, zc_e = encoder(gen_imgs_samp, seval=True)
                lat_mse_loss = mse_loss(zn_e, zn_samp)
                lat_xe_loss = xe_loss(zc_e, zc_samp_idx)
                c_zn.append(lat_mse_loss.item())
                c_zc.append(lat_xe_loss.item())
                
                # Save some examples!
                save_image(r_imgs.data[:25], '%s/real_%06i.png' %(run_dir, batches_done), 
                           nrow=5, normalize=True)
                save_image(reg_imgs.data[:25], '%s/reg_%06i.png' %(run_dir, batches_done), 
                           nrow=5, normalize=True)
                save_image(gen_imgs.data[:25], '%s/gen_%06i.png' %(run_dir, batches_done), 
                           nrow=5, normalize=True)
                
                d_l.append(d_loss.item())
                ge_l.append(ge_loss.item())
                g_l.append(g_loss.item())
                e_l.append(e_loss.item())
    
    
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] "\
                       "[GE loss: %f] [G loss: %f] [E loss: %f]" % ( epoch, 
                                                                     n_epochs, 
                                                                     i, 
                                                                     len(dataloader),
                                                                     d_loss.item(),
                                                                     ge_loss.item(),
                                                                     g_loss.item(), 
                                                                     e_loss.item())
                      )
                
                print("\t Cycle Losses: [img %f] [latn %f] [latc %f]"%(img_mse_loss.item(), 
                                                                       lat_mse_loss.item(), 
                                                                       lat_xe_loss.item()))


    # Rudimentary plotting section
    earr = range(0, len(d_l))
    
    fig, ax = plt.subplots(figsize=(16,10))
    ax.plot(earr, d_l, label='Dis')
    ax.plot(earr, ge_l, label='Gen+Enc')
    ax.plot(earr, g_l, label='Gen')
    ax.plot(earr, e_l, label='Enc')
    
    ax.set(xlabel='Epoch', ylabel='Loss',
           title='Loss vs Training Epoch')
    #ax.set_xlim(230, 260)
    #ax.set_xlim(0, 10)
    #ax.set_ylim(-0.01, 40.3)
    ax.grid()
    #plt.yscale('log')
    plt.legend(loc='upper right', fontsize=16)
    
    fig.savefig("%s/gan_loss_training.png"%(run_dir))
    
    
    # Data for plotting
    earr = range(0, len(d_l))#n_epochs)
    
    fig, ax = plt.subplots(figsize=(16,10))
    ax.plot(earr, c_i, label='R')
    ax.plot(earr, c_zn, label=r'$Z_n$')
    ax.plot(earr, c_zc, label=r'$Z_c$')
    
    ax.set(xlabel='Epoch', ylabel='MSE',
           title='Reco MSE vs Training Epoch')
    #ax.set_xlim(230, 260)
    #ax.set_xlim(0, 10)
    #ax.set_ylim(-0.01, 40.3)
    ax.grid()
    plt.yscale('log')
    plt.legend(loc='upper right', fontsize=16)
    
    fig.savefig("%s/reco_mse_training.png"%(run_dir))

    # Save current state of gen and disc models
    disc_fname = "%s/discriminator_MNIST.pth.tar"%run_dir
    gen_fname = "%s/generator_MNIST.pth.tar"%run_dir
    enc_fname = "%s/encoder_MNIST.pth.tar"%run_dir
    torch.save(discriminator.state_dict(), disc_fname)
    torch.save(generator.state_dict(), gen_fname)
    torch.save(encoder.state_dict(), enc_fname)


if __name__ == "__main__":
    main()
