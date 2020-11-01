import argparse

import os

import torch
from IPython.core.debugger import set_trace
from torch import nn
from torch.nn import functional as F
from data import data_helper
# from IPython.core.debugger import set_trace
from data.data_helper import available_datasets
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler
from utils.Logger import Logger
import numpy as np
import time
import pkbar
import json
import os.path as osp
from PIL import Image  
import pickle


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=225, help="Image size")
    # data aug stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.0, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.0, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscaling a tile")
    #
    parser.add_argument("--limit_source", default=None, type=int, help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int, help="If set, it will limit the number of testing samples")

    parser.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=31, help="Number of classes")
    parser.add_argument("--jigsaw_n_classes", "-jc", type=int, default=31, help="Number of classes for the jigsaw task")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use", default="caffenet")
    parser.add_argument("--jig_weight", type=float, default=0.1, help="Weight for the jigsaw puzzle")
    parser.add_argument("--ooo_weight", type=float, default=0, help="Weight for odd one out task")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default=None, help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=None, type=float, help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", action='store_true', help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", action='store_true',
                        help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", action='store_true', help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", action='store_true', help="Use nesterov")
    
    parser.add_argument("--jig_only", action = "store_true", help = "Disable classification loss")

    return parser.parse_args()



class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        model = model_factory.get_network(args.network)(jigsaw_classes=args.jigsaw_n_classes + 1, classes=args.n_classes)
        self.model = model.to(device)
        # print(self.model)
        #self.source_loader, self.val_loader = data_helper.get_train_dataloader(args, patches=model.is_patch_based())
        self.target_loader = data_helper.get_val_dataloader(args, patches=model.is_patch_based())
        #self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        #self.len_dataloader = len(self.source_loader)
        print("Dataset size: test %d" % , (len(self.target_loader.dataset)))
        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, args.train_all, nesterov=args.nesterov)
        self.jig_weight = args.jig_weight
        self.only_non_scrambled = args.classify_only_sane
        self.n_classes = args.n_classes
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None

        self.best_val_jigsaw = 0.0
        self.best_jigsaw_acc = 0.0
        folder_name, logname = Logger.get_name_from_args(args)
        

        self.folder_name = folder_name
        #self.save_folder = os.path.join("logs", folder_name, logname)

    def hook(self, model, input, output):
        output = output.view(output.size(0), -1)
        print(output.size())
        self.activations.append(output.detach().cpu().numpy())

    def get_features(self):
        loader = self.target_loader
        model_path = osp.join('logs',self.folder_name,'best_model.pth')
        self.activations = []
        sd = torch.load(model_path)
        self.model.load_state_dict(sd['model_state_dict'])
        self.model.avgpool.register_forward_hook(self.hook)
        self.model.eval()
        pbar = pkbar.Pbar(name='Epoch Progress',target=len(loader))
        labels = []
        for it, ((data, jig_l, class_l), _) in enumerate(loader):
            pbar.update(it)
            data, jig_l, class_l = data.to(self.device), jig_l.to(self.device), class_l.to(self.device)
            jigsaw_logit, class_logit = self.model(data)
            labels.append(class_l.detach().cpu().numpy())
        target_data = {}
        target_data['activations'] = self.activations
        target_data['labels'] = labels
        with open(osp.join('logs',self.folder_name, 'act_labels.pkl'), 'wb') as handle:
          pickle.dump(target_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return self.activations 
      

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.get_features()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
