import copy
import pandas as pd
from torchvision import datasets, transforms
from dataset.mnist import MnistDataset
from client import Client
from topology import adj_list_gen
from model.create_model import create_mnist_model, create_cifar_model
from .dataprocess import data_processing


def create_clients(train_data_path, test_data_path, alpha = 0.4, client_num = 10, task = "mnist",
                   model_type = "simple", topo = "ring", batch_size = 128, lr = 0.8,
                   gamma = 0.8, device = 'cuda', pseu_agg = False):
    """
    Prams:
        ...
    Return:
        List of Client().
    """

    adj_list = adj_list_gen(client_num, topo)

    for i in range(client_num):
        print("Neighbor of ", i, ":", adj_list[i])

    if task == "mnist" or task == "fashion-mnist":
        ini_model = create_mnist_model(model_type = model_type)
        client_datas = data_processing(train_data_path, alpha, client_num, "mnist")
        test_data = pd.read_csv(test_data_path)
        test_dataset = MnistDataset(test_data)
    elif task == "cifar10":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        ini_model = create_cifar_model(model_type = model_type)
        client_datas = data_processing(train_data_path, alpha, client_num, "cifar10")
        test_dataset = datasets.CIFAR10(root=train_data_path, train=False, download=True,
                                        transform=transform)
    else:
        raise ValueError("In create_clients(): task is not approriate!")


    list_client = [Client(i, copy.deepcopy(ini_model), client_datas[i], test_dataset,
                        batch_size = batch_size, lr = lr, gamma = gamma, device = device,
                        pseu_agg = pseu_agg) for i in range(client_num)]

    for i in range(client_num):
        for j in adj_list[i]:
            list_client[i].list_neighbor.append(list_client[j])

            ### For pseu_agg
            if pseu_agg:
                list_client[i].list_model_neighbor.append(
                    copy.deepcopy(list_client[j].model.state_dict()))
                n_model_dict = copy.deepcopy(list_client[j].model.state_dict())
                for key in n_model_dict.keys():
                    n_model_dict[key] *= 0
                list_client[i].ema_delta.append(n_model_dict)

    return list_client
