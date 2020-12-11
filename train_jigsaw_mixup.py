import warnings
warnings.simplefilter(action = "ignore")

import argparse
import os
import random
import torch
from IPython.core.debugger import set_trace
from torch import nn
from torch.nn import functional as F
from data import data_helper
from data import mixup_data
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


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=225, help="Image size")
    # data aug stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.0, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.0, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float,
                        help="Chance of randomly greyscaling a tile")
    #
    parser.add_argument("--limit_source", default=None, type=int,
                        help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int,
                        help="If set, it will limit the number of testing samples")

    parser.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=31, help="Number of classes")
    parser.add_argument("--jigsaw_n_classes", "-jc", type=int, default=31, help="Number of classes for the jigsaw task")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use",
                        default="caffenet")
    parser.add_argument("--jig_weight", type=float, default=0.1, help="Weight for the jigsaw puzzle")
    parser.add_argument("--ooo_weight", type=float, default=0, help="Weight for odd one out task")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default=None, help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=None, type=float,
                        help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", action='store_true', help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", action='store_true',
                        help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", action='store_true', help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", action='store_true', help="Use nesterov")

    parser.add_argument("--jig_only", action="store_true", help="Disable classification loss")
    parser.add_argument("--stylized", action = "store_true", help = "Use txt_files/StylizedPACS/")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--dataset", choices = ['PACS', 'OfficeHome'], help="Dataset Name sued for training")
    parser.add_argument("--mix_up", action = "store_true", help = "Using Domain MixUp")

    return parser.parse_args()


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        model = model_factory.get_network(args.network)(jigsaw_classes=args.jigsaw_n_classes + 1,
                                                        classes=args.n_classes)
        self.model = model.to(device)
        # print(self.model)
        #self.source_loader, self.val_loader = mixup_data.get_mixup_domain_data(args,['art'])
        self.source_loader, self.val_loader = mixup_data.get_mixup_dataloader(args, patches=model.is_patch_based())
        self.val_loader = self.source_loader
        self.target_loader = data_helper.get_jigsaw_val_dataloader(args, patches=model.is_patch_based())
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, val %d, test %d" % (
        len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))
        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, args.train_all,
                                                                 nesterov=args.nesterov)
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
        _, logname = Logger.get_name_from_args(args)

        self.folder_name = "%s/%s_to_%s/%s" % (args.folder_name, 
            "-".join(sorted(args.source)), args.target, logname)

    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    

    def _do_epoch(self):
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        epoch_loss = 0
        pbar = pkbar.Pbar(name = 'Epoch Progress', target = len(self.source_loader))
        for it, (data, jig_l, class_x, class_y, lam) in enumerate(self.source_loader):
            pbar.update(it)
            data, jig_l, class_x, class_y, lam = data.to(self.device), jig_l.to(self.device),class_x.to(self.device), class_y.to(
                self.device), lam.to(self.device)
            # absolute_iter_count = it + self.current_epoch * self.len_dataloader
            # p = float(absolute_iter_count) / self.args.epochs / self.len_dataloader
            # lambda_val = 2. / (1. + np.exp(-10 * p)) - 1
            # if domain_error > 2.0:
            #     lambda_val  = 0
            # print("Shutting down LAMBDA to prevent implosion")

            self.optimizer.zero_grad()
            jigsaw_logit, class_logit = self.model(data)  # , lambda_val=lambda_val)

   
            jigsaw_loss = criterion(jigsaw_logit, jig_l)
            # domain_loss = criterion(domain_logit, d_idx)
            # domain_error = domain_loss.item()
            if self.args.mix_up:
                class_loss = self.mixup_criterion(criterion,class_logit, class_x, class_y, lam) 
            

            _, cls_pred = class_logit.max(dim=1)
            _, jig_pred = jigsaw_logit.max(dim=1)
            # _, domain_pred = domain_logit.max(dim=1)
            if self.args.jig_only:
                class_loss = torch.Tensor([0.0])
                loss = jigsaw_loss
            # _, domain_pred = domain_logit.max(dim=1)
            else:
                loss = class_loss + jigsaw_loss * self.jig_weight  # + 0.1 * domain_loss
            epoch_loss = epoch_loss + loss
            loss.backward()
            self.optimizer.step()

            self.logger.log(it, len(self.source_loader),
                            {"jigsaw": jigsaw_loss.item(), "class": class_loss.item()  # , "domain": domain_loss.item()
                             },
                            # ,"lambda": lambda_val},
                            {"jigsaw": torch.sum(jig_pred == jig_l.data).item(),
                             "class": 0
                             # "domain": torch.sum(domain_pred == d_idx.data).item()
                             },
                            data.shape[0])
            del loss, class_loss, jigsaw_loss, jigsaw_logit, class_logit

        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                if phase == 'val':
                    jigsaw_correct, class_correct = self.do_val(loader)
                else:
                    jigsaw_correct, class_correct = self.do_test(loader)
                jigsaw_acc = float(jigsaw_correct) / total
                class_acc = float(class_correct) / total
                #class_acc = 0
                self.logger.log_test(phase, {"jigsaw": jigsaw_acc, "class": class_acc})
                self.results[phase][self.current_epoch] = jigsaw_acc

        if (self.results['val'][self.current_epoch] > self.best_jigsaw_acc):
            self.best_jigsaw_acc = self.results['val'][self.current_epoch]
            print("Saving new best at epoch: {}".format(self.current_epoch))
            self.save_model(os.path.join("logs", self.folder_name, 'best_model.pth'))

        print("Saving latest at epoch: {}".format(self.current_epoch))
        self.save_model(os.path.join("logs", self.folder_name, 'latest_model.pth'))

    def save_model(self, file_path):
        torch.save({'epoch': self.current_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': self.results['val'][self.current_epoch],
                    'test_acc': self.results['test'][self.current_epoch]}, file_path)

    def do_val(self, loader):
        jigsaw_correct = 0
        class_correct = 0
        domain_correct = 0
        for it,(data, jig_l, class_x, class_y, lam) in enumerate(loader):
            data, jig_l, class_x, class_y = data.to(self.device), jig_l.to(self.device), class_x.to(self.device), class_y.to(self.device)

            jigsaw_logit, class_logit = self.model(data)
            #_, cls_pred = class_logit.max(dim=1)
            _, jig_pred = jigsaw_logit.max(dim=1)
            class_correct = 0
            jigsaw_correct += torch.sum(jig_pred == jig_l.data)
        return jigsaw_correct, class_correct
    
    def do_test(self, loader):
        jigsaw_correct = 0
        class_correct = 0
        domain_correct = 0
        for it, ((data, jig_l, class_l), _) in enumerate(loader):
            data, jig_l, class_l = data.to(self.device), jig_l.to(self.device), class_l.to(self.device)

            jigsaw_logit, class_logit = self.model(data)
            _, cls_pred = class_logit.max(dim=1)
            _, jig_pred = jigsaw_logit.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l.data)
            jigsaw_correct += torch.sum(jig_pred == jig_l.data)
        return jigsaw_correct, class_correct

    def do_test_multi(self, loader):
        jigsaw_correct = 0
        class_correct = 0
        single_correct = 0
        for it, ((data, jig_l, class_l), d_idx) in enumerate(loader):
            data, jig_l, class_l = data.to(self.device), jig_l.to(self.device), class_l.to(self.device)
            n_permutations = data.shape[1]
            class_logits = torch.zeros(n_permutations, data.shape[0], self.n_classes).to(self.device)
            for k in range(n_permutations):
                class_logits[k] = F.softmax(self.model(data[:, k])[1], dim=1)
            class_logits[0] *= 4 * n_permutations  # bias more the original image
            class_logit = class_logits.mean(0)
            _, cls_pred = class_logit.max(dim=1)
            jigsaw_logit, single_logit = self.model(data[:, 0])
            _, jig_pred = jigsaw_logit.max(dim=1)
            _, single_logit = single_logit.max(dim=1)
            single_correct += torch.sum(single_logit == class_l.data)
            class_correct += torch.sum(cls_pred == class_l.data)
            jigsaw_correct += torch.sum(jig_pred == jig_l.data[:, 0])
        return jigsaw_correct, class_correct, single_correct

    def do_training(self):
        self.logger = Logger(self.args, update_frequency=30)  # , "domain", "lambda"
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}

        for self.current_epoch in range(self.args.epochs):
            start_time = time.time()
            self.scheduler.step()
            self.logger.new_epoch(self.scheduler.get_lr())
            self._do_epoch()
            end_time = time.time()
            print(f"Runtime of the epoch is {end_time - start_time}")
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        print(
            "Best val %g, corresponding test %g - best test: %g" % (val_res.max(), test_res[idx_best], test_res.max()))
        self.logger.save_best(test_res[idx_best], test_res.max())

        # Save Arguments
        with open(osp.join('logs', self.folder_name, 'args.txt'), 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)

        return self.logger, self.model


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    trainer = Trainer(args, device)
    trainer.do_training()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
