import torch

def select_device(device='GPU', batch_size=None):

    """ Select correct device, i.e. CPU or (multi) GPU """

    # Ensure you have CPU if you selected it
    cuda = (device == 'GPU' and torch.cuda_is_available)

    return torch.device('cuda:0' if cuda else 'cpu')