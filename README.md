# ClusterGAN: A PyTorch Implementation

This is a PyTorch implementation of [ClusterGAN](https://arxiv.org/abs/1809.03627),
an approach to unsupervised clustering using generative adversarial networks.





## Requirements

The package as well as the necessary requirements can be installed by running `make` or via
```
python setup.py install
```

## Run ClusterGAN on MNIST

To run ClusterGAN on the MNIST dataset, ensure the package is setup and then run
```
python train_mnist.py -r test_run
```
where a directory `runs/test_run` will be made and contain the generated output from the training run.

## License

[MIT License](LICENSE)

Copyright (c) 2018 Zigfried Hampel-Arias
