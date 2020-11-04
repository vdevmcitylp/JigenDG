import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from PIL import Image
from torch.utils.data import Dataset
import torch.optim as optim
import timeit
import pickle
import os.path as osp


class LinearEvalDataset(Dataset):
    
    def __init__(self, pkl_file, root_dir, train_split=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        data = self.load_file(root_dir+pkl_file)
        if train_split:
          count = int(len(data['labels'])*train_split)
        else:
          count = len(data['labels'])
        self.features = data['features'][:count]
        self.labels = data['labels'][:count]
        self.root_dir = root_dir
        self.transform = transform

    def load_file(self, pkl_file):
      with open(pkl_file, 'rb') as handle:
          data = pickle.load(handle)
      return data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        feature = self.features[idx]
        label = self.labels[idx]
        sample = (feature, label)
        if self.transform:
            sample = self.transform(sample)
        return sample


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, test_acc

def shuffle(x, y):
    perm = np.random.permutation(len(y))
    x = x[perm]
    y = y[perm]
    return x,y

def split_data(root, split):
    pkl_file = osp.join(root, 'act_labels.pkl')
    with open(pkl_file, 'rb') as handle:
          data = pickle.load(handle)
    n = int(split*len(data['labels']))
    features = data['features']
    labels = data['labels']
    features, labels = shuffle(features, labels)
    tr_data = {}
    te_data = {}
    tr_data['features'] = features[:n]
    tr_data['labels'] = labels[:n]
    te_data['features'] = features[n:]
    te_data['labels'] = labels[n:]

    with open(osp.join(root, 'train_data.pkl'), 'wb') as handle:
          pickle.dump(tr_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(root, 'test_data.pkl'), 'wb') as handle:
          pickle.dump(te_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def LinearEval(root, split=None):
    
    train_dataset = LinearEvalDataset('train_data.pkl', root, train_split=split)
    test_dataset = LinearEvalDataset('test_data.pkl', root)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

    model = nn.Linear(512,7).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_acc = 0
    start = timeit.default_timer()
    for epoch in range(1, 100):
        train(model, device, train_loader, criterion, optimizer, epoch)
        loss, acc = test(model, device, test_loader, criterion)
        if acc > best_acc:
          best_acc = acc
        stop = timeit.default_timer()
    print('Total time taken: {} seconds'.format(int(stop - start)) )
    print('Best Acc: {}'.format(best_acc))
    
    
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    #Location of the train and test pkl files - with 50-50 split
    #Use split data function to create the splits
    data_root = '/content/drive/My Drive/Codes/JigenDG/logs/cartoon_target_stylizedjigsaw/art-photo-sketch_to_cartoon/'
    split = 0.1
    LinearEval(data_root, split=split)
