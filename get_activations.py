import warnings
warnings.simplefilter(action = "ignore")
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
import functools

def get_args():
    
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=225, help="Image size")
    
    parser.add_argument("--limit_source", default=None, type=int, help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int, help="If set, it will limit the number of testing samples")

    parser.add_argument("--n_classes", "-c", type=int, default=31, help="Number of classes")
    parser.add_argument("--jigsaw_n_classes", "-jc", type=int, default=31, help="Number of classes for the jigsaw task")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use", default="caffenet")

    parser.add_argument("--exp_type")#, choices = ["vanilla-jigsaw", "stylized-jigsaw"])
        
    parser.add_argument("--stylized", action = "store_true", help = "Use txt_files/StylizedPACS/")
    parser.add_argument("--run_id", type = str, help = "Run ID of the experiment, model will be loaded from and \
        activations will be saved to args.exp_type/s1-s2-s3_to_s4/args.run_id")

    parser.add_argument("--generate_for", type = str, help = "Generate activations for this domain")

    parser.add_argument("--dataset", choices = ["PACS", "OfficeHome"])

    return parser.parse_args()

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        model = model_factory.get_network(args.network)(jigsaw_classes=args.jigsaw_n_classes + 1, 
            classes=args.n_classes)
        self.model = model.to(device)
    
        self.exp_type = "%s/%s_to_%s/%s" % (args.exp_type, 
            "-".join(sorted(args.source)), args.target, args.run_id)

        self.model_path = osp.join('logs', self.exp_type, 'best_model.pth')
        
        print("Loading best_model.pth from {}".format(self.model_path))
        
        self.args.target = self.args.generate_for

        self.target_loader = data_helper.get_val_dataloader(args, 
            patches=model.is_patch_based())

        print("Dataset size: test %d" % (len(self.target_loader.dataset)))

        self.n_classes = args.n_classes
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None
        


    def hook(self, model, input, output):
        output = output.view(output.size(0), -1)
        print(output.size())
        self.activations.append(output.detach().cpu().numpy())

    def get_features(self):
        loader = self.target_loader
        self.activations = []
        sd = torch.load(self.model_path)
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
        self.activations = functools.reduce((lambda x,y: np.append(x,y,axis=0)),self.activations)
        labels = functools.reduce((lambda x,y: np.append(x,y,axis=0)),labels)
        
        target_data = {}
        target_data['features'] = self.activations
        target_data['labels'] = labels
        with open(osp.join('logs', self.exp_type, 'act_labels_{}.pkl'.format(self.args.target)), 'wb') as handle:
          pickle.dump(target_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Activations saved to logs/{}/act_labels_{}.pkl".format(self.exp_type, self.args.target))

        return self.activations 
      

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.get_features()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
