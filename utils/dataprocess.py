import torch
import pandas as pd
import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms
from dataset.mnist import MnistDataset
from .datadis import non_iid_partition_with_dirichlet_distribution, sort_and_partition


def data_processing(data_path, alpha = 1.0, client_num = 20, data_type = 'mnist'):
    """
    Param:
        data_path: path of training file
        alpha: alpha in Dirichlet distribution
        client_num: number of clients
        data_type: Current options: "mnist".
    Return:
        List of MnistDataset coresponding to each of client
    """
    if data_type == 'mnist':
        return data_processing_for_mnist(data_path, alpha, client_num, 10)
    elif data_type == "cifar10":
        return data_processing_for_cifar(data_path, alpha, client_num, 10)
    raise SyntaxError("In data_processing(): data_type is not approriate")

def data_processing_for_mnist(csv_path, alpha = 1.0, client_num = 20, num_class = 10):
    data = pd.read_csv(csv_path)
    label = torch.tensor(data['label'].values)

    if alpha >= 0:
        dis_sample = non_iid_partition_with_dirichlet_distribution(label_list = label,
                                                                   client_num = client_num,
                                                                   classes = num_class,
                                                                   alpha = alpha)
    else:
        dis_sample = sort_and_partition(label_list = label,
                                        client_num = client_num,
                                        classes = num_class)

    client_datas = []
    for i in range(client_num):
        client_datas.append(MnistDataset(data = data, sample_mask = dis_sample[i]))

    return client_datas


def data_processing_for_cifar(train_data_path, alpha = 1.0, client_num = 20, num_class = 10):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = datasets.CIFAR10(root=train_data_path, train=True, download=True, transform=transform)
    label = np.array(dataset.targets)
    if alpha >= 0:
        dis_sample = non_iid_partition_with_dirichlet_distribution(label_list = label,
                                                                   client_num = client_num,
                                                                   classes = num_class,
                                                                   alpha = alpha)
    else:
        dis_sample = sort_and_partition(label_list = label,
                                        client_num = client_num,
                                        classes = num_class)

    client_datas = []
    for client_id in range(client_num):
        indices = dis_sample[client_id]
        subset = Subset(dataset, indices)
        client_datas.append(subset)

    return client_datas
