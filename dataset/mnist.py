import torch
from torch.utils.data import Dataset

class MnistDataset(Dataset):
    def __init__(self, data, sample_mask = None, transform=None, target_transform=None):
        """
        Dataset class for Mnist
        Parameters:
            data: dataframe read from pd.read_csv
            sample_mask: List of index that can actually be used
        """
        self.sample_mask = sample_mask # list
        self.img=torch.tensor(data.drop(columns='label').values).reshape((data.shape[0],28,28))/255
        self.label = torch.tensor(data['label'].values)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        if self.sample_mask is None:
            return len(self.img)
        return len(self.sample_mask)

    def __getitem__(self, idx):
        if self.sample_mask is None:
            image = self.img[idx]
            label = self.label[idx]
        else:
            image = self.img[self.sample_mask[idx]]
            label = self.label[self.sample_mask[idx]]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
