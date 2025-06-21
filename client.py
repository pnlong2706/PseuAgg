import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class Client():
    def __init__(self, idx, model, train_dataset, test_dataset, **kargs):
        self.id = idx
        if ('device' in kargs and torch.cuda.is_available() and kargs['device'] == 'cuda'):
            self.device = 'cuda'
        else:
            self.device = "cpu"
        self.model = model.to(self.device)
        self.agg_model = model
        self.list_neighbor = []   ## Ini in create_client !!!
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=kargs['batch_size'],
            shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1000,
            shuffle=False
        )
        self.data_len = len(train_dataset)

        ### Directly modify optimizer and scheduler here
        self.optim = optim.Adadelta(self.model.parameters(), lr=kargs['lr'])
        self.scheduler = StepLR(self.optim, step_size=4, gamma=kargs['gamma'])

        ### For pseudo aggregation
        self.enable_pseu_agg = kargs['pseu_agg']
        self.beta_agg = 0.98
        self.w_agg = 1.0
        self.list_model_neighbor = [] # List of state_dict, Ini in create_client !!!
        self.ema_delta = [] ## Ini in create_client !!!

    def train(self, epoch, log = True):
        self.model.train()
        device = self.device
        avg_loss = 0
        total_batch = 1
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(device), target.to(device)
            self.optim.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            avg_loss += loss.item()
            total_batch = batch_idx + 1
            loss.backward()
            self.optim.step()

        if log:
            print(f"Epoch {epoch:<3}. Client {self.id:<2}. Average training loss: \
                {(avg_loss / total_batch):.4f}")

        avg_loss /= total_batch
        return avg_loss

    def test(self, epoch, log = True):
        self.model.eval()
        device = self.device
        with torch.no_grad():
            test_loss = 0
            correct = 0
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(self.test_loader.dataset)

            if log:
                print(f"Epoch {epoch:<3}. Client {self.id:<2}. Average test loss: {test_loss:.4f},\
                      Test accuracy: {correct}/{len(self.test_loader.dataset)} \
                      ({(100. * correct / len(self.test_loader.dataset)):.0f}%)")

            return test_loss, 100. * correct / len(self.test_loader.dataset)

    def weighted_avg(self, agg_model, agg_weight, n_model_states, n_weight):
        """
        Params:
            agg_model: state_dict of the model to be aggregated
            agg_weight: weight for agg_model in avg, eg. len of dataset
            n_model_states: list of state_dict of neighbor model
            n_weight: list of weight coresponding to n_model_states
        Return:
            aggragated model's state_dict
        """

        with torch.no_grad():
            total_sum = agg_weight
            for ele in n_weight:
                total_sum += ele
            for key in agg_model.keys():
                agg_model[key] = torch.div(agg_model[key], (total_sum / agg_weight))
                # pylint: disable=consider-using-enumerate
                for i in range(len(n_model_states)):
                    agg_model[key] += n_model_states[i][key] * (n_weight[i] / total_sum)

        return agg_model

    def aggregate(self):
        self.agg_model = copy.deepcopy(self.model.state_dict())
        with torch.no_grad():
            n_model_states = [client.model.state_dict() for client in self.list_neighbor]
            n_weights = [client.data_len for client in self.list_neighbor]

            ## For pseu_agg
            if self.enable_pseu_agg:
                for key in self.agg_model.keys():
                    # pylint: disable=consider-using-enumerate
                    for i in range(len(self.ema_delta)):
                        self.ema_delta[i][key] = self.ema_delta[i][key]*0.75 + \
                        (n_model_states[i][key]-self.list_model_neighbor[i][key])*0.25
                        self.list_model_neighbor[i][key] = n_model_states[i][key]

            self.agg_model = self.weighted_avg(
                self.agg_model,
                self.data_len,
                n_model_states,
                n_weights
            )

    # The reason why we need to do this here instead of in aggragate is to make sure that
    # after the aggregation of node i, node j can still take node i's old model (model before
    # aggregation), This happen cuz we simulate using for loop for sync comm, not real life sim or
    # multi thread or async comm.
    def update_aggregate(self):
        with torch.no_grad():
            self.model.load_state_dict(self.agg_model)

    def pseu_aggregate(self):
        with torch.no_grad():
            if not self.enable_pseu_agg:
                return

            n_weights = [client.data_len * self.w_agg for client in self.list_neighbor]
            n_model_states = [{} for i in range(len(self.list_model_neighbor))]
            self.agg_model = copy.deepcopy(self.model.state_dict())

            for key in self.agg_model.keys():
                # pylint: disable=consider-using-enumerate
                for i in range(len(self.list_model_neighbor)):
                    n_model_states[i][key] = self.list_model_neighbor[i][key] + \
                        self.ema_delta[i][key] * 0.5

            self.agg_model = self.weighted_avg(
                self.agg_model,
                self.data_len,
                n_model_states,
                n_weights
            )

            self.w_agg *= self.beta_agg
            self.model.load_state_dict(self.agg_model)
