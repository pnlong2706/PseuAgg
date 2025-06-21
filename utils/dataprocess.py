import torch
import pandas as pd

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
