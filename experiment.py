import copy
import torch
import numpy as np

from dfl import DFL

def experiment(name, args, **kwargs):
    """
    Params:
        args: initial config, like above
        **kwargs: values to be adjusted in args
            You can change all value of keyword already in args
            Add `visualize_data = i` to visualize data of client i (0-index)
    """
    print("###########", name, "###########\n")

    c_args = copy.deepcopy(args)
    for key, value in kwargs.items():
        c_args[key] = value

    torch.manual_seed(c_args['seed'])
    np.random.seed(c_args['seed'])

    print("Config:",c_args, "\n")

    dfl = DFL(**c_args)
    print("Preparation done...")

    if 'visualize_data' in kwargs:
        print("Visualize data distribution on client",kwargs['visualize_data'])
        dfl.visualize_data_dis(kwargs['visualize_data'])

    dfl.training()

    print("\n------------------------------")

