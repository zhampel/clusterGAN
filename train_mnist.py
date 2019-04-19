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
    from clusgan.utils import tlog, save_model, calc_gradient_penalty, sample_z, cross_entropy
    from clusgan.plots import plot_train_loss
except ImportError as e:
    print(e)
    raise ImportError

def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default='clusgan', help="Name of training run")
    args = parser.parse_args()

    # Make directory structure for this run
    run_name = args.run_name
    run_dir = '%s/%s'%(RUNS_DIR, run_name)
    imgs_dir = '%s/images'%(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(imgs_dir, exist_ok=True)
    print('\nResults to be saved in directory %s\n'%(run_dir))
    
    n_epochs = 200
    batch_size = 64
    lr = 1e-4
    b1 = 0.5
    b2 = 0.9 #99
    decay = 2.5*1e-5
    img_size = 28
    channels = 1
    n_skip_iter = 5
   
    # Latent space info
    latent_dim = 30
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
   
    ge_chain = ichain(generator.parameters(),
                      encoder.parameters())
    optimizer_GE = torch.optim.Adam(ge_chain, lr=lr, betas=(b1, b2), weight_decay=decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    #optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=decay)
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

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
           
            # Zero gradients for models
            generator.zero_grad()
            encoder.zero_grad()
            discriminator.zero_grad()
            # Ensure generator is trainable
            generator.train()
            
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
            zn, zc, zc_idx = sample_z(shape=imgs.shape[0], latent_dim=latent_dim, n_c=n_c, req_grad=False)
    
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
                    ge_loss = -torch.mean(tlog(D_gen)) + betan * zn_loss + betac * zc_loss
    
                ge_loss.backward(retain_graph=True)
                #ge_loss.backward()
                optimizer_GE.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
    
            optimizer_D.zero_grad()
    
            # Measure discriminator's ability to classify real from generated samples
            if wass_metric:
                # Gradient penalty term
                grad_penalty = calc_gradient_penalty(discriminator, real_imgs, gen_imgs)
                #grad_penalty.backward(retain_graph=True)

                # Wasserstein GAN loss w/gradient penalty
                d_loss = torch.mean(D_real) - torch.mean(D_gen) + grad_penalty
                
            else:
                # Vanilla GAN loss
                d_loss = -torch.mean(tlog(D_real) - tlog(1 - D_gen))
    
            d_loss.backward()
            optimizer_D.step()


        # Generator in eval mode
        generator.eval()

        # Set number of examples for cycle calcs
        n_sqrt_samp = 5
        n_samp = n_sqrt_samp * n_sqrt_samp

        # Cycle through real -> enc -> gen
        r_imgs, i_label = real_imgs.data[:n_samp], itruth_label[:n_samp]
        e_zn, e_zc, e_zc_logits = encoder(r_imgs, seval=True)
        reg_imgs = generator(e_zn, e_zc)
        img_mse_loss = mse_loss(r_imgs, reg_imgs)
        # Save img reco cycle loss
        c_i.append(img_mse_loss.item())
        
        # Cycle through enc -> gen -> enc
        gen_imgs_idx = []
        stack_imgs = []
        for idx in range(n_c):
            zn_samp, zc_samp, zc_samp_idx = sample_z(shape=10, latent_dim=latent_dim, n_c=n_c, fix_class=idx, req_grad=False)
            gen_imgs_samp = generator(zn_samp, zc_samp)
            zn_e, zc_e, zc_e_logits = encoder(gen_imgs_samp, seval=True)
            if (len(stack_imgs) == 0):
                stack_imgs = gen_imgs_samp
            else:
                stack_imgs = torch.cat((stack_imgs, gen_imgs_samp), 0)
            #print("Sampled: ", zc_samp_idx)
            #print("Encoded: ", torch.argmax(zc_e, dim=1))
            #print("Loss: ", xe_loss(zc_e_logits, zc_samp_idx))
            gen_imgs_idx.append(gen_imgs_idx)

        save_image(stack_imgs, '%s/gen_classes_%06i.png' %(imgs_dir, epoch), 
                   nrow=10, normalize=True)
      

        zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_samp, latent_dim=latent_dim, n_c=n_c, req_grad=False)
        gen_imgs_samp = generator(zn_samp, zc_samp)
        zn_e, zc_e, zc_e_logits = encoder(gen_imgs_samp, seval=True)
        #print("Sampled: ", zc_samp_idx)
        #print("Encoded: ", torch.argmax(zc_e, dim=1))
        #print("Loss: ", xe_loss(zc_e_logits, zc_samp_idx))
        lat_mse_loss = mse_loss(zn_e, zn_samp)
        lat_xe_loss = xe_loss(zc_e_logits, zc_samp_idx)
        #lat_xe_loss = cross_entropy(zc_e_logits, zc_samp)
        # Save latent space cycle losses
        c_zn.append(lat_mse_loss.item())
        c_zc.append(lat_xe_loss.item())
      

        # Save some examples!
        save_image(r_imgs.data[:n_samp], '%s/real_%06i.png' %(imgs_dir, epoch), 
                   nrow=n_sqrt_samp, normalize=True)
        save_image(reg_imgs.data[:n_samp], '%s/reg_%06i.png' %(imgs_dir, epoch), 
                   nrow=n_sqrt_samp, normalize=True)
        save_image(gen_imgs_samp.data[:n_samp], '%s/gen_%06i.png' %(imgs_dir, epoch), 
                   nrow=n_sqrt_samp, normalize=True)
        
        # Save training losses
        d_l.append(d_loss.item())
        ge_l.append(ge_loss.item())
    
    
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
                             'gen_enc_loss' : ['G+E', ge_l],
                             'disc_loss' : ['D', d_l],
                             'zn_cycle_loss' : ['$||Z_n-E(G(x))_n||$', c_zn],
                             'zc_cycle_loss' : ['$||Z_c-E(G(x))_c||$', c_zc],
                             'img_cycle_loss' : ['$||X-G(E(x))||$', c_i]
                            })


    # Plot some training results
    plot_train_loss(df=train_df,
                    arr_list=['gen_enc_loss', 'disc_loss'],
                    figname='%s/gan_training_loss.png'%(run_dir)
                    )

    plot_train_loss(df=train_df,
                    arr_list=['zn_cycle_loss', 'zc_cycle_loss', 'img_cycle_loss'],
                    figname='%s/cycle_training_loss.png'%(run_dir)
                    )


    # Save current state of gen and disc models
    save_model(model=discriminator, name="%s/discriminator_MNIST.pth.tar"%(run_dir))
    save_model(model=encoder, name="%s/encoder_MNIST.pth.tar"%(run_dir))
    save_model(model=generator, name="%s/generator_MNIST.pth.tar"%(run_dir))


if __name__ == "__main__":
    main()
