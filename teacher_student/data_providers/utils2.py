#from .cifar import Cifar10DataProvider, Cifar100DataProvider, \
#    Cifar10AugmentedDataProvider, Cifar100AugmentedDataProvider
from .cifar2 import Cifar100DataProvider2, Cifar10DataProvider2
#from .svhn import SVHNDataProvider
from .mnist2 import MnistDataProvider


def get_data_provider_by_name(name, train_params):
    """Return required data provider class"""
    if name == 'C10':
        return Cifar10DataProvider2(**train_params)
    #if name == 'C10+':
    #    return Cifar10AugmentedDataProvider(**train_params)
    if name == 'C100':
        return Cifar100DataProvider2(**train_params)
    #if name == 'C100+':
    #    return Cifar100AugmentedDataProvider(**train_params)
    #if name == 'SVHN':
    #    return SVHNDataProvider(**train_params)
    if name == 'MNIST':
        return MnistDataProvider(**train_params)
    else:
        print("Sorry, data provider for `%s` dataset "
              "was not implemented yet" % name)
        exit()
