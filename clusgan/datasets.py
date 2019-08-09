from __future__ import print_function

try:
    import numpy as np
    
    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets
    import torchvision.transforms as transforms
except ImportError as e:
    print(e)
    raise ImportError


DATASET_FN_DICT = {'mnist' : datasets.MNIST,
                   'fashion-mnist' : datasets.FashionMNIST
                  }


dataset_list = DATASET_FN_DICT.keys()


def get_dataset(dataset_name='mnist'):
    """
    Convenience function for retrieving
    allowed datasets.
    Parameters
    ----------
    name : {'mnist', 'fashion-mnist'}
          Name of dataset
    Returns
    -------
    fn : function
         PyTorch dataset
    """
    if dataset_name in DATASET_FN_DICT:
        fn = DATASET_FN_DICT[dataset_name]
        return fn
    else:
        raise ValueError('Invalid dataset, {}, entered. Must be '
                         'in {}'.format(dataset_name, DATASET_FN_DICT.keys()))



def get_dataloader(dataset_name='mnist', data_dir='', batch_size=64, train_set=True, num_workers=1):

    dset = get_dataset(dataset_name)

    dataloader = torch.utils.data.DataLoader(
        dset(data_dir, train=train_set, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True)

    return dataloader
