from re import X
import scipy.io
import numpy as np
import torch
import mne

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from sklearn.preprocessing import OrdinalEncoder
# from sklearn.preprocessing import OneHotEncode

class dataset(Dataset):
    def __init__(self, X, y, train=True):
        self.X = X
        self.y = y
        self.train=train

    def __len__(self):
        return len(self.y)

    # def __getitem__(self, idx):
    #     rng = np.random.randint(0, high=200)
    #     if self.train:
    #         x = self.X[idx][:, rng:rng + 600]
    #     else:
    #         x = self.X[idx][:, 200: 800]
    #     return x, self.y[idx]

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.train:
#             rn = np.random.randint(0, high=500)
#             x = x[:, rn:rn+4000]
            x = x[:, 0:4000]
        else:
            x = x[:, 0:4000]
        return x, self.y[idx]


class dataset_with_domain(Dataset):
    def __init__(self, X, y, y_domain, train=True):
        self.X = X
        self.y = y
        self.y_domain = y_domain
        self.train=train

    def __len__(self):
        return len(self.y)

    # def __getitem__(self, idx):
    #     rng = np.random.randint(0, high=200)
    #     if self.train:
    #         x = self.X[idx][:, rng:rng + 600]
    #     else:
    #         x = self.X[idx][:, 200: 800]
    #     return x, self.y[idx]

    def __getitem__(self, idx):
        # idx = 1000
        x = self.X[idx]
        if self.train:
#             rn = np.random.randint(0, high=500)
#             x = x[:, rn:rn+4000]
            x = x[:, 0:4000]
        else:
            x = x[:, 0:4000]
        return x, self.y[idx], self.y_domain[idx]


def split_data(X, y,train_size=1800,val_size=200,test_size=304):
    def get_idx():
        np.random.seed(seed=42)
        # rng = np.random.choice(len(y), len(y), replace=False)
        rng = np.random.choice(y.shape[0], y.shape[0], replace=False)
        return rng
    # train_size, val_size, test_size = 1000, 197, 200
    # train_size, val_size, test_size = 1800, 200, 304      ### test data    这里参数需要修改，测试数据小批量这么多
    indices = get_idx()
    train_idx, val_idx, test_idx = indices[0: train_size], \
                indices[train_size: train_size + val_size], indices[train_size + val_size:]
    train_X, train_y, val_X, val_y, test_X, test_y = \
        X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]
    print("Finish split")
    return train_X, train_y, val_X, val_y, test_X, test_y

def split_data_with_domain(X, y_label, y_domain_label, train_size=1800, val_size=200, test_size=304):
    def get_idx():
        np.random.seed(seed=42)
        # rng = np.random.choice(len(y), len(y), replace=False)
        rng = np.random.choice(y_label.shape[0], y_label.shape[0], replace=False)
        return rng
    indices = get_idx()
    train_idx, val_idx, test_idx = indices[0: train_size], \
                                   indices[train_size: train_size + val_size], indices[train_size + val_size:]
    train_X, train_y, train_domain_y, val_X, val_y, val_domain_y, test_X, test_y, test_domain_y = \
        X[train_idx], y_label[train_idx], y_domain_label[train_idx], \
        X[val_idx], y_label[val_idx], y_domain_label[val_idx], \
        X[test_idx], y_label[test_idx], y_domain_label[test_idx]
    print("Finish split")
    return train_X, train_y, train_domain_y, val_X, val_y, test_X, test_y


def get_loaders(train_X, train_y, val_X, val_y, test_X, test_y,batch_size = 250):
    train_set, val_set, test_set = dataset(train_X, train_y, True), dataset(val_X, val_y, False), dataset(test_X, test_y, False)
    data_loader_train = torch.utils.data.DataLoader(
        train_set, 
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True, 
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
            val_set, 
            batch_size=batch_size,
            num_workers=1,
            pin_memory=True, 
            drop_last=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
            test_set, 
            batch_size=batch_size,
            num_workers=1,
            pin_memory=True, 
            drop_last=False,
    )
    dataloaders = {
        'train': data_loader_train,
        'val': data_loader_val,
        'test': data_loader_test
    }
    return dataloaders

def get_loaders_with_domain(train_X, train_y, train_domain_y, test_X, test_y, test_domain_y, batch_size = 250):
    train_set, test_set = dataset_with_domain(train_X, train_y, train_domain_y, True), \
                                   dataset_with_domain(test_X, test_y, test_domain_y, False)
    data_loader_train = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=0,
        # pin_memory=True,
        drop_last=False,
        shuffle= True
    )
    data_loader_test = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            num_workers=0,
            # pin_memory=True,
            drop_last=False,
            shuffle=False
    )
    dataloaders = {
        'train': data_loader_train,
        'test': data_loader_test
    }
    return dataloaders

