import json
import os
import torch
import matplotlib.pyplot as plt
from utils.client import create_clients

class DFL:
    def __init__(self, **kwargs):
        self.args = kwargs

        device = 'cpu'
        if self.args['device'] == 'cuda' and torch.cuda.is_available():
            device = 'cuda'

        task_name = self.args['task']

        if task_name == 'mnist':
            train_path = self.args['data_path'] + 'mnist_train.csv'
            test_path = self.args['data_path'] + 'mnist_test.csv'
        elif task_name == 'fashion-mnist':
            train_path = self.args['data_path'] + 'fashion-mnist_test.csv'
            test_path = self.args['data_path'] + 'fashion-mnist_train.csv'
        else:
            raise SyntaxError("Task name is not approriate!")

        alpha_dis = self.args['alpha_dis']
        client_num = self.args['client_num']

        self.clients = create_clients(train_path,
                                 test_path,
                                 alpha_dis,
                                 model_type = self.args['model_type'],
                                 client_num = client_num,
                                 task = task_name,
                                 topo = self.args['topo'],
                                 batch_size = self.args['batch_size'],
                                 lr = self.args['lr'],
                                 gamma = self.args['gamma'],
                                 device = device,
                                 pseu_agg = self.args['pseu_agg'])


    def training(self, max_epoch = -1):
        res = [self.args,]
        client_num = self.args['client_num']
        max_accu_all = 0

        if max_epoch == -1:
            max_epoch = self.args['epochs']

        for epoch in range(1,max_epoch+1):
            avg_loss = 0
            avg_test_loss = 0
            max_accu = 0

            for idx in range(client_num):
                avg_loss += self.clients[idx].train(epoch, log = self.args['client_train_log'])

            if self.args['agg'] and epoch%self.args['agg_epoch'] == 0:
                for idx in range(client_num):
                    self.clients[idx].aggregate()
                for idx in range(client_num):
                    self.clients[idx].update_aggregate()

            elif self.args['pseu_agg'] and epoch%self.args['agg_epoch']==self.args['agg_epoch']//2 \
            and epoch > self.args['agg_epoch']:
                for idx in range(client_num):
                    self.clients[idx].pseu_aggregate()

            for idx in range(client_num):
                x, y = self.clients[idx].test(epoch, log = self.args['client_train_log'])
                avg_test_loss += x
                max_accu = max(max_accu, y)

            avg_loss /= client_num
            avg_test_loss /= client_num
            max_accu_all = max(max_accu_all, max_accu)
            if epoch%self.args['log_interval'] == 0:
                print(f"Epoch {epoch:<3}. Average training loss: {avg_loss:.4f}. ", end="")
                print(f"Average testing loss: {avg_test_loss:.4f}. Max accuracy: {max_accu}")

            res.append({'train_loss':avg_loss, 'test_loss': avg_test_loss, 'max_accu': max_accu})

        print("MAX ACCURACY:", max_accu_all)

        res_path = self.args['save_path'] + self.args['result_file']
        if os.path.exists(self.args['save_path']):
            overwrite = self.args['overwrite_res']
            while os.path.exists(res_path) and not overwrite:
                print("Path to save statistic result is already exist, overwrite it? Y/N")
                ans = input()
                if ans == 'Y':
                    overwrite = True
                elif ans == 'N':
                    print("Enter new file name (.json):")
                    new_file_name = input()
                    res_path = self.args['save_path'] + new_file_name

            # pylint: disable=unspecified-encoding
            with open(res_path, 'w+') as fout:
                json.dump(res, fout)

    def visualize_data_dis(self,i):
        """
        Input: client index to visualize its data distribution
        """
        cnt = [0 for i in range(10)]
        client = self.clients[i]
        for _, target in client.train_loader:
            for label in target:
                cnt[label] += 1
        plt.bar([ "'"+str(i)+"'" for i in range(10)],cnt)
        plt.show()
