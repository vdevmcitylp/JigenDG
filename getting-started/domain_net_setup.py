"""
Adds /DATA1/vaasudev_narayanan/datasets/DomainNet/ to every line
"""
import pdb
import os.path as osp


def main(string_to_add, domains, root):

	for domain in domains:
		for mode in ["train", "test"]:
			file_name = osp.join(root, "{}_{}.txt".format(domain, mode))
			
			with open(file_name, 'r') as f:
				file_lines = [''.join([string_to_add, x.strip(), '\n']) for x in f.readlines()]

			with open(file_name, 'w') as f:
				f.writelines(file_lines)


if __name__ == '__main__':
	
	root = osp.join("..", "data", "txt_lists", "VanillaDomainNet")
	string_to_add = "/DATA1/vaasudev_narayanan/datasets/DomainNet/"
	domains = ("clipart", "painting", "quickdraw", "real", "sketch")	

	main(string_to_add, domains, root)