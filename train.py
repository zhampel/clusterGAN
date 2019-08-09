from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np

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
    from clusgan.utils import save_model, calc_gradient_penalty, sample_z, cross_entropy
    from clusgan.datasets import get_dataloader, dataset_list
    from clusgan.plots import plot_train_loss
except ImportError as e:
    print(e)
    raise ImportError

def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default='clusgan', help="Name of training run")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=200, type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("-s", "--dataset_name", dest="dataset_name", default='mnist', choices=dataset_list,  help="Dataset name")
    parser.add_argument("-w", "--wass_metric", dest="wass_metric", action='store_true', help="Flag for Wasserstein metric")
    parser.add_argument("-g", "-–gpu", dest="gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument("-k", "-–num_workers", dest="num_workers", default=1, type=int, help="Number of dataset workers")
    args = parser.parse_args()

    run_name = args.run_name
    dataset_name = args.dataset_name
    device_id = args.gpu
    num_workers = args.num_workers

    # Training details
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    test_batch_size = 5000
    lr = 1e-4
    b1 = 0.5
    b2 = 0.9 #99
    decay = 2.5*1e-5
    n_skip_iter = 1 #5

    img_size = 28
    channels = 1
   
    # Latent space info
    latent_dim = 30
    n_c = 10
    betan = 10
    betac = 10
   
    # Wasserstein metric flag
    # Wasserstein metric flag
    wass_metric = args.wass_metric
    mtype = 'van'
    if (wass_metric):
        mtype = 'wass'
    
    # Make directory structure for this run
    sep_und = '_'
    run_name_comps = ['%iepoch'%n_epochs, 'z%s'%str(latent_dim), mtype, 'bs%i'%batch_size, run_name]
    run_name = sep_und.join(run_name_comps)

    run_dir = os.path.join(RUNS_DIR, dataset_name, run_name)
    data_dir = os.path.join(DATASETS_DIR, dataset_name)
    imgs_dir = os.path.join(run_dir, 'images')
    models_dir = os.path.join(run_dir, 'models')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    print('\nResults to be saved in directory %s\n'%(run_dir))
    
    x_shape = (channels, img_size, img_size)
    
    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device_id)

    # Loss function
    bce_loss = torch.nn.BCELoss()
    xe_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    
    # Initialize generator and discriminator
    generator = Generator_CNN(latent_dim, n_c, x_shape)
    encoder = Encoder_CNN(latent_dim, n_c)
    discriminator = Discriminator_CNN(wass_metric=wass_metric)
    
    if cuda:
        generator.cuda()
        encoder.cuda()
        discriminator.cuda()
        bce_loss.cuda()
        xe_loss.cuda()
        mse_loss.cuda()
        
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    # Configure training data loader
    dataloader = get_dataloader(dataset_name=dataset_name,
                                data_dir=data_dir,
                                batch_size=batch_size,
                                num_workers=num_workers)

    # Test data loader
    testdata = get_dataloader(dataset_name=dataset_name, data_dir=data_dir, batch_size=test_batch_size, train_set=False)
    test_imgs, test_labels = next(iter(testdata))
    test_imgs = Variable(test_imgs.type(Tensor))
   
    ge_chain = ichain(generator.parameters(),
                      encoder.parameters())
    optimizer_GE = torch.optim.Adam(ge_chain, lr=lr, betas=(b1, b2), weight_decay=decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    #optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=decay)

    # ----------
    #  Training
    # ----------
    ge_l = []
    d_l = []
    
    c_zn = []
    c_zc = []
    c_i = []
    
    # Training loop 
    print('\nBegin training session with %i epochs...\n'%(n_epochs))
    for epoch in range(n_epochs):
        for i, (imgs, itruth_label) in enumerate(dataloader):
           
            # Ensure generator/encoder are trainable
            generator.train()
            encoder.train()
            # Zero gradients for models
            generator.zero_grad()
            encoder.zero_grad()
            discriminator.zero_grad()
            
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------------
            #  Train Generator + Encoder
            # ---------------------------
            
            optimizer_GE.zero_grad()
            
            # Sample random latent variables
            zn, zc, zc_idx = sample_z(shape=imgs.shape[0],
                                      latent_dim=latent_dim,
                                      n_c=n_c)
    
            # Generate a batch of images
            gen_imgs = generator(zn, zc)
            
            # Discriminator output from real and generated samples
            D_gen = discriminator(gen_imgs)
            D_real = discriminator(real_imgs)
            
            # Step for Generator & Encoder, n_skip_iter times less than for discriminator
            if (i % n_skip_iter == 0):
                # Encode the generated images
                enc_gen_zn, enc_gen_zc, enc_gen_zc_logits = encoder(gen_imgs)
    
                # Calculate losses for z_n, z_c
                zn_loss = mse_loss(enc_gen_zn, zn)
                zc_loss = xe_loss(enc_gen_zc_logits, zc_idx)
                #zc_loss = cross_entropy(enc_gen_zc_logits, zc)
    
                # Check requested metric
                if wass_metric:
                    # Wasserstein GAN loss
                    ge_loss = torch.mean(D_gen) + betan * zn_loss + betac * zc_loss
                else:
                    # Vanilla GAN loss
                    valid = Variable(Tensor(gen_imgs.size(0), 1).fill_(1.0), requires_grad=False)
                    v_loss = bce_loss(D_gen, valid)
                    ge_loss = v_loss + betan * zn_loss + betac * zc_loss
    
                ge_loss.backward(retain_graph=True)
                optimizer_GE.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
    
            optimizer_D.zero_grad()
    
            # Measure discriminator's ability to classify real from generated samples
            if wass_metric:
                # Gradient penalty term
                grad_penalty = calc_gradient_penalty(discriminator, real_imgs, gen_imgs)

                # Wasserstein GAN loss w/gradient penalty
                d_loss = torch.mean(D_real) - torch.mean(D_gen) + grad_penalty
                
            else:
                # Vanilla GAN loss
                fake = Variable(Tensor(gen_imgs.size(0), 1).fill_(0.0), requires_grad=False)
                real_loss = bce_loss(D_real, valid)
                fake_loss = bce_loss(D_gen, fake)
                d_loss = (real_loss + fake_loss) / 2
    
            d_loss.backward()
            optimizer_D.step()


        # Save training losses
        d_l.append(d_loss.item())
        ge_l.append(ge_loss.item())
   

        # Generator in eval mode
        generator.eval()
        encoder.eval()

        # Set number of examples for cycle calcs
        n_sqrt_samp = 5
        n_samp = n_sqrt_samp * n_sqrt_samp


        ## Cycle through test real -> enc -> gen
        t_imgs, t_label = test_imgs.data, test_labels
        #r_imgs, i_label = real_imgs.data[:n_samp], itruth_label[:n_samp]
        # Encode sample real instances
        e_tzn, e_tzc, e_tzc_logits = encoder(t_imgs)
        # Generate sample instances from encoding
        teg_imgs = generator(e_tzn, e_tzc)
        # Calculate cycle reconstruction loss
        img_mse_loss = mse_loss(t_imgs, teg_imgs)
        # Save img reco cycle loss
        c_i.append(img_mse_loss.item())
       

        ## Cycle through randomly sampled encoding -> generator -> encoder
        zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_samp,
                                                 latent_dim=latent_dim,
                                                 n_c=n_c)
        # Generate sample instances
        gen_imgs_samp = generator(zn_samp, zc_samp)
        # Encode sample instances
        zn_e, zc_e, zc_e_logits = encoder(gen_imgs_samp)
        # Calculate cycle latent losses
        lat_mse_loss = mse_loss(zn_e, zn_samp)
        lat_xe_loss = xe_loss(zc_e_logits, zc_samp_idx)
        #lat_xe_loss = cross_entropy(zc_e_logits, zc_samp)
        # Save latent space cycle losses
        c_zn.append(lat_mse_loss.item())
        c_zc.append(lat_xe_loss.item())
      
        # Save cycled and generated examples!
        r_imgs, i_label = real_imgs.data[:n_samp], itruth_label[:n_samp]
        e_zn, e_zc, e_zc_logits = encoder(r_imgs)
        reg_imgs = generator(e_zn, e_zc)
        save_image(r_imgs.data[:n_samp],
                   '%s/real_%06i.png' %(imgs_dir, epoch), 
                   nrow=n_sqrt_samp, normalize=True)
        save_image(reg_imgs.data[:n_samp],
                   '%s/reg_%06i.png' %(imgs_dir, epoch), 
                   nrow=n_sqrt_samp, normalize=True)
        save_image(gen_imgs_samp.data[:n_samp],
                   '%s/gen_%06i.png' %(imgs_dir, epoch), 
                   nrow=n_sqrt_samp, normalize=True)
        
        ## Generate samples for specified classes
        stack_imgs = []
        for idx in range(n_c):
            # Sample specific class
            zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_c,
                                                     latent_dim=latent_dim,
                                                     n_c=n_c,
                                                     fix_class=idx)

            # Generate sample instances
            gen_imgs_samp = generator(zn_samp, zc_samp)

            if (len(stack_imgs) == 0):
                stack_imgs = gen_imgs_samp
            else:
                stack_imgs = torch.cat((stack_imgs, gen_imgs_samp), 0)

        # Save class-specified generated examples!
        save_image(stack_imgs,
                   '%s/gen_classes_%06i.png' %(imgs_dir, epoch), 
                   nrow=n_c, normalize=True)
     

        print ("[Epoch %d/%d] \n"\
               "\tModel Losses: [D: %f] [GE: %f]" % (epoch, 
                                                     n_epochs, 
                                                     d_loss.item(),
                                                     ge_loss.item())
              )
        
        print("\tCycle Losses: [x: %f] [z_n: %f] [z_c: %f]"%(img_mse_loss.item(), 
                                                             lat_mse_loss.item(), 
                                                             lat_xe_loss.item())
             )

    


    # Save training results
    train_df = pd.DataFrame({
                             'n_epochs' : n_epochs,
                             'learning_rate' : lr,
                             'beta_1' : b1,
                             'beta_2' : b2,
                             'weight_decay' : decay,
                             'n_skip_iter' : n_skip_iter,
                             'latent_dim' : latent_dim,
                             'n_classes' : n_c,
                             'beta_n' : betan,
                             'beta_c' : betac,
                             'wass_metric' : wass_metric,
                             'gen_enc_loss' : ['G+E', ge_l],
                             'disc_loss' : ['D', d_l],
                             'zn_cycle_loss' : ['$||Z_n-E(G(x))_n||$', c_zn],
                             'zc_cycle_loss' : ['$||Z_c-E(G(x))_c||$', c_zc],
                             'img_cycle_loss' : ['$||X-G(E(x))||$', c_i]
                            })

    train_df.to_csv('%s/training_details.csv'%(run_dir))


    # Plot some training results
    plot_train_loss(df=train_df,
                    arr_list=['gen_enc_loss', 'disc_loss'],
                    figname='%s/training_model_losses.png'%(run_dir)
                    )

    plot_train_loss(df=train_df,
                    arr_list=['zn_cycle_loss', 'zc_cycle_loss', 'img_cycle_loss'],
                    figname='%s/training_cycle_loss.png'%(run_dir)
                    )


    # Save current state of trained models
    model_list = [discriminator, encoder, generator]
    save_model(models=model_list, out_dir=models_dir)


if __name__ == "__main__":
    main()
