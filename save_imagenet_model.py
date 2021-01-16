""" 
Saves a ImageNet pre-trained model 
to for eg. PACS/vanilla-jigsaw/m-m-m_to_m/0/best_model.pth
"""

import os
import random
import argparse

import torch
import numpy as np

from models import model_factory

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

	parser = argparse.ArgumentParser()
	
	parser.add_argument("--network", choices = model_factory.nets_map.keys(), help = "Which network to use",
                        default = "caffenet")
	parser.add_argument("--n_classes", "-c", type = int, default = 31, help = "Number of classes")
	parser.add_argument("--jigsaw_n_classes", "-jc", type = int, default = 31, 
		help = "Number of classes for the jigsaw task")

	parser.add_argument("--folder_name", help = "Top level folder, eg. PACS/vanilla-jigsaw")

	args = parser.parse_args()

	return args


def main():

	args = get_args()

	model = model_factory.get_network(args.network)(jigsaw_classes = args.jigsaw_n_classes + 1, 
		classes = args.n_classes)

	save_path = os.path.join("logs", args.folder_name, "m-m-m-m_to_m", "0")
	
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	save_path = os.path.join(save_path, "best_model.pth")

	torch.save({"model_state_dict": model.state_dict()}, save_path)


if __name__ == '__main__':
	
	main()