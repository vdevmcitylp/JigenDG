import os
import os.path as osp
import shutil

import pdb

def main(num_samples, dataset, domains):

	root = osp.join("/DATA1", "vaasudev_narayanan", "repositories", "JigenDG", "data", "txt_lists")
	dataset_root = osp.join(root, "Stylized{}".format(dataset))

	destination_root = osp.join(root, "Stylized{}_{}".format(dataset, num_samples))

	for dom_as_target in domains:

		target_folder = osp.join(destination_root, "{}_target".format(dom_as_target))

		if not osp.exists(target_folder):
			os.makedirs(target_folder)

		for domain in domains:
			for mode in ["train", "test"]:
				
				file_name = osp.join(dataset_root, "{}_target".format(dom_as_target), 
					"{}_{}.txt".format(domain, mode))
				
				destination_file_name = osp.join(target_folder, "{}_{}.txt".format(domain, mode))
				
				if domain != dom_as_target:

					f_in = open(file_name, "r")
					f_out = open(destination_file_name, "w")
					# pdb.set_trace()
					for line in f_in:
						line = "{}PACS_{}{}".format(line.split("PACS")[0], num_samples, line.split("PACS")[1])
						for n in range(num_samples):
							sample_line = "{}_s{}.{}".format(line.split(".")[0], n, line.split(".")[1])
							f_out.write(sample_line)
				else:
					shutil.copy(file_name, destination_file_name)


if __name__ == '__main__':
	
	num_samples = 3
	dataset = "PACS"
	domains = ("photo", "art", "cartoon", "sketch")

	main(num_samples, dataset, domains)