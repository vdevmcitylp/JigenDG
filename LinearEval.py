import argparse
import random
import pandas as pd
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
import pkbar

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

class Model3(nn.Module):
    def __init__(self, n_classes):
        super(Model3, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(512,512,3,1,1,bias=False)
        self.conv2 = nn.Conv2d(512,512,3,1,1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, n_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
class Model2(nn.Module):

    def __init__(self, n_classes):
        super(Model2, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(512,512,3,1,1,bias=False)
        self.bn = nn.BatchNorm2d(512)
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        # Max pooling over a (2, 2) windo
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x) 
        return x

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
    #pbar =pkbar.Pbar(name = "Epoch Progress", target = len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
     #   pbar.update(batch_idx)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        # if batch_idx % 100 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))


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
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    return test_loss, test_acc

def shuffle(x, y):
    perm = np.random.permutation(len(y))
    x = x[perm]
    y = y[perm]
    return x,y

def split_data(root, split, target, layer):
    pkl_file = osp.join(root, 'act_labels_{}_layer_{}.pkl'.format(target, layer))
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


def get_model(layer, n_classes):
    if layer == 1:
        model = nn.Linear(512, n_classes).cuda()
    elif layer == 2:
        model = Model2(n_classes)
    elif layer == 3:
        model = Model3(n_classes)
    return model

def LinearEval(root, n_classes, layer, split=None):
    
    train_dataset = LinearEvalDataset('train_data.pkl', root, train_split=split)
    test_dataset = LinearEvalDataset('test_data.pkl', root)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

    model = get_model(layer, n_classes).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_acc = 0
    start = timeit.default_timer()
    for epoch in range(1, 100):
      #  print("Training epoch {}/100".format(epoch))
        train(model, device, train_loader, criterion, optimizer, epoch)
        loss, acc = test(model, device, test_loader, criterion)
        if acc > best_acc:
          best_acc = acc
        stop = timeit.default_timer()
    # print('Total time taken: {} seconds'.format(int(stop - start)) )
    # print('Best Acc: {}'.format(best_acc))
    
    return best_acc
    
def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help = "Source", nargs = '+')
    parser.add_argument("--target", help = "Target")
    parser.add_argument("--freeze_layer", default=1, type=int, help="Number of layers to train")

    parser.add_argument("--exp_type")#, choices = ["vanilla-jigsaw", "stylized-jigsaw"])
    parser.add_argument("--run_id", type = str, help = "Run ID of the experiment, act_label.pkl \
        will be loaded from args.exp_type/s1-s2-s3_to_s4/args.run_id")

    parser.add_argument("--n_classes", type = int, choices = [7, 65])

    parser.add_argument("--seed", type = int)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    
    args = get_args()
    set_seed(args.seed)

    # "vanilla-jigsaw/art-photo-sketch_to_cartoon/6068/" 
    exp_folder = "%s/%s_to_%s/%s/" % (args.exp_type, 
            "-".join(sorted(args.source)), args.target, args.run_id) 

    logs_root = "/DATA1/neha_t/JiGen/logs"
    logs_folder = osp.join(logs_root, exp_folder)
    
    # Splitting target domain (logs_folder/act_label.pkl first to get 50-50 split)
    # Saves train_data.pkl and test_data.pkl in the same logs_folder
    split_data(logs_folder, split = 0.5, target = args.target, layer=args.freeze_layer)
    
    splits = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1}

    best_accs_dict = {}

    #for split in splits:
    #    best_split_acc = LinearEval(logs_folder, args.n_classes, split = split)
    #    best_accs_dict[split] = best_split_acc
    best_accs_dict = {}
    for split in splits:
       print("Running for split: ",split)
       best_split_acc = LinearEval(logs_folder, args.n_classes, args.freeze_layer, split = split)
       best_accs_dict[split] = round(best_split_acc * 100, 2)

    df = pd.DataFrame.from_dict(best_accs_dict, orient = "index")
    print(df.to_csv(sep = '\t', index = False))
    #print(best_accs_dict)

    # python LinearEval.py --source art photo sketch --target cartoon --exp_type vanilla-jigsaw --run_id 6068
    # Assumes act_label.pkl is stored at logs_root/vanilla-jigsaw/art-photo-sketch_to_cartoon/6068
