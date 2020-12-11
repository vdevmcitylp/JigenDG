from os.path import join, dirname
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import sample, random
from data import StandardDataset
from data.JigsawLoader import JigsawDataset, JigsawTestDataset, get_split_dataset_info, _dataset_info, JigsawTestDatasetMultiple
from data.concat_dataset import ConcatDataset
from data.DomainDataLoader import DomainDataset

def get_train_transformers(args):
    img_tr = [transforms.RandomResizedCrop(int(args.image_size), (args.min_scale, args.max_scale))]
    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))
    img_tr = img_tr + [transforms.ToTensor()]
    tile_tr = []
    if args.tile_random_grayscale:
        tile_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))
    tile_tr = tile_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr), transforms.Compose(tile_tr)

def get_domain_dataloader(args,domains):
    dataset_list = list(domains)
    datasets = []
    val_datasets = []
    img_transformer, tile_transformer = get_train_transformers(args)

    for dname in dataset_list:
        name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'txt_lists', 'Vanilla'+ args.dataset ,'%s_train.txt' % dname), args.val_size)
        
        train_dataset = DomainDataset(name_train, labels_train, img_transformer=img_transformer)

        datasets.append(train_dataset)
        val_datasets.append(DomainDataset(name_val, labels_val, img_transformer=img_transformer))
        
    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader, val_loader

def mixup_data(x_data, x_label, y_data, y_label,alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x_data.size()[0]
    mixed_data = lam * x_data + (1 - lam) * y_data[:batch_size]
    return mixed_data, x_label, y_label[:batch_size], lam



def get_mixup_dataloader(args, patches=True):
    img_transformer, tile_transformer = get_train_transformers(args)
    print('Loading Train Data') 
    tr_dataset = JigsawMixUpDataset(args, jig_classes = args.jigsaw_n_classes, tile_transformer=tile_transformer, patches = patches, train=True)
    tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    print('Loading Val Data')
    val_dataset = JigsawMixUpDataset(args, jig_classes = args.jigsaw_n_classes, tile_transformer=tile_transformer, patches = patches, train=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    return tr_loader, val_loader

def get_domain_dataloader(args, domains):
    dataset_list = list(domains)
    datasets = []
    val_datasets = []
    img_transformer, tile_transformer = get_train_transformers(args)

    for dname in dataset_list:
        name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'txt_lists', 'Vanilla'+ args.dataset ,'%s_train.txt' % dname), args.val_size)
          
        train_dataset = DomainDataset(name_train, labels_train, img_transformer=img_transformer)

        datasets.append(train_dataset)
        val_datasets.append(DomainDataset(name_val, labels_val, img_transformer=img_transformer))
            
    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader, val_loader

class JigsawMixUpDataset(data.Dataset):
    def __init__(self, args, jig_classes = 100, tile_transformer=None,patches=True, train=True):
        self.source_list = args.source
        self.args = args
        self.permutations = self.__retrieve_permutations(jig_classes)
        self.grid_size = 3
        self.train = train
        self.bias_whole_image = args.bias_whole_image
        if patches:
            self.patch_size = 64
        self._augment_tile = tile_transformer
        if patches:
            self.returnFunc = lambda x: x
        else:
            def make_grid(x):
                return torchvision.utils.make_grid(x, self.grid_size, padding=0)
            self.returnFunc = make_grid
        
        # New code
        self.images = torch.Tensor([])
        self.labels_x = torch.Tensor([])
        self.labels_y = torch.Tensor([])
        self.lam = []
        self.makeMixUpData(self.source_list, self.args)

   

    def mix_up_domains(self, dom_x, dom_y):
        dom_y_iter = iter(dom_y)
        for it, ((data, class_l), _) in enumerate(dom_x):
            batch_x_data = data
            batch_x_label = class_l
            batch_y_data, batch_y_label = dom_y_iter.next()
            mixed_data, batch_x_label, batch_y_label, lam = mixup_data(batch_x_data, batch_x_label, batch_y_data[0], batch_y_label)
            if len(self.lam) > 0:
              self.images = self.images+mixed_data
              self.labels_x = self.labels_x + batch_x_label
              self.labels_y = self.labels_y + batch_y_label
              self.lam = self.lam + [lam]
            else:
              self.images = mixed_data
              self.labels_x = batch_x_label
              self.labels_y = batch_y_label
              self.lam = [lam]
      

    def makeMixUpData(self, source_list, args):
      
        start_time = time.time()
        for dom_x in source_list:
            print('Mixing Data '+ dom_x)
            dom_y = [i for i in source_list if i != dom_x]
            dom_x_trloader, dom_x_valloader = get_domain_dataloader(self.args, [dom_x])
            dom_y_trloader, dom_y_valloader = get_domain_dataloader(self.args, dom_y)
            if self.train:
                self.mix_up_domains(dom_x_trloader, dom_y_trloader)
            else:
                self.mix_up_domains(dom_x_valloader, dom_y_valloader)
        end_time = time.time()
        print(f"Time taken for domain mixup is {end_time - start_time}") 

    def get_tile(self, img, n):
        
        w = float(img.size[0]) / self.grid_size
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        tile = self._augment_tile(tile)
        return tile
    
    def __getitem__(self, index):
        img = self.images[index]
        topil = transforms.ToPILImage()
        img = topil(img)
        n_grids = self.grid_size ** 2
        tiles = [None] * n_grids
        for n in range(n_grids):
            tiles[n] = self.get_tile(img, n)
        order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
        if self.bias_whole_image:
            if self.bias_whole_image > random():
                order = 0
        if order == 0:
            data = tiles
        else:
            data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]
            
        data = torch.stack(data, 0)

        x_label = self.labels_x[index]
        y_label = self.labels_x[index]
        lam = self.lam[int(index/self.args.batch_size)]
        fin_im = self.returnFunc(data)
        return self.returnFunc(data), int(order), x_label, y_label, lam

    def __len__(self):
        return len(self.labels_x)
    
    def __retrieve_permutations(self, classes):
        all_perm = np.load('permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1
        return all_perm
        
        









